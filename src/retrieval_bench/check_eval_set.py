"""Validate an eval set against the blog posts and report coverage and word overlap.

Checks the schema (the Question model below), post slugs, and that each required
source post has an evidence quote found in that post. A found quote shows the post
has the text, not that the reference answer is right.

Also reports, per question, how much of its wording appears in its source posts,
weighted by word rarity, since rare words drive BM25. Words are counted after
retrieval.tokenize (BM25's stopwords and stemmer), so the number reflects what
BM25 sees. Overlap is measured, not controlled: the notebook plots retrieval
results against it.

Items with a blank `question` are skipped, so the v2 workbook (eval-set-v2.yaml)
can be checked too.

Usage: uv run check-eval-set [--file dataset/eval-set-v2-shortlist.yaml] [--blog PATH]
                             [--post-cap 3] [--overlap]
       uv run check-eval-set --explain cs04    # per-term breakdown for one question
"""
import argparse
import collections
import math
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from retrieval_bench.retrieval import tokenize

Slug = str  # a post's filename without .mdx, e.g. "my-homelab"
Term = str  # one token after retrieval.tokenize: lowercased, stopwords dropped, stemmed

QuestionType = Literal["single_fact", "multi_post", "current_state", "opinion", "unanswerable"]
Topic = Literal["ai-llm", "ml-nlp", "backend", "web-design", "homelab", "self-hosting",
                "computing", "personal", "writing-productivity", "cross-topic"]
Probe = Literal["vocab_mismatch", "lexical_collision", "sibling_confusion", "near_miss",
                "caption_only", "alt_text_only", "table_only", "code_only", "callout_only",
                "buried_detail", "short_post", "long_post", "recency", "false_premise",
                "aggregation", "multi_hop"]

POST_FIELDS = ("source_posts", "acceptable_posts", "superseded_posts", "related_posts")
DEFAULT_BLOG = Path.home() / "Projects/personal-blog-astro/src/content/blog"
ROOT = Path(__file__).resolve().parents[2]  # project root (src/retrieval_bench/ is two levels down)

NAME = {"bryan"}  # after tokenizing, "Bryan's" becomes "bryan"


class Question(BaseModel):
    """One eval-set item, as written in the YAML. The field meanings are in the
    YAML file's header comment.

    Parsing checks each field on its own: required fields are present, `type`,
    `topic` and `probe` use an allowed value, and lists hold strings. validate()
    checks the rules that span fields or need the posts.
    """
    model_config = ConfigDict(extra="forbid")  # a misspelled field name is an error

    id: str
    type: QuestionType
    topic: Topic
    probe: Probe | None = None
    question: str
    source_posts: list[Slug] = []
    acceptable_posts: list[Slug] = []
    superseded_posts: list[Slug] = []
    related_posts: list[Slug] = []
    evidence: dict[Slug, str] = {}
    reference_answer: str = Field(min_length=1)
    key_facts: list[str] = []
    notes: str | None = None

    # Reference only; not used by any check
    v1_question: str | None = None
    v1_paraphrase: str | None = None
    paraphrase: str | None = None
    in_v1_subset: bool = False


def norm(s: str) -> str:
    s = s.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"}))
    s = s.replace("\\", "").replace("`", "").replace("**", "")
    return re.sub(r"\s+", " ", s).lower().strip()

def strip_links(s: str) -> str:
    return re.sub(r"\[([^\]]*)\]\(\s*<?[^)\s>]*>?\s*\)", r"\1", s)

def contains(raw: str, quote: str) -> bool:
    q = norm(quote)
    return q in norm(raw) or q in norm(strip_links(raw))

def load_posts(blog: str | Path = DEFAULT_BLOG) -> dict[Slug, str]:
    """slug -> the post's raw MDX text."""
    return {p.stem: p.read_text(encoding="utf-8") for p in Path(blog).glob("*.mdx")}

def load_items(path: str | Path) -> list[dict[str, Any]]:
    """Every item in the file as a plain dict, blank `question` or not."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["questions"]

def is_blank(item: dict[str, Any]) -> bool:
    """A blank `question` marks a workbook item that wasn't chosen."""
    return not (item.get("question") or "").strip()

def parse_questions(items: list[dict[str, Any]]) -> tuple[list[Question], list[str]]:
    """Parse each item into a Question. Returns the ones that parsed, and one error
    message per problem in the ones that didn't."""
    questions, errors = [], []
    for item in items:
        try:
            questions.append(Question.model_validate(item))
        except ValidationError as e:
            qid = item.get("id", "?")
            for err in e.errors():
                field = ".".join(str(part) for part in err["loc"])
                errors.append(f"{qid}: {field}: {err['msg']}")
    return questions, errors


class TermRow(NamedTuple):
    """One line of WordOverlap.explain()."""
    term: Term
    idf: float
    found: dict[str, bool]  # group label, such as "current", -> is the term in those posts?


class WordOverlap:
    """How much of a question's wording appears in a given set of posts, as BM25 sees it.

        overlap = sum of IDF over question terms that appear in the posts
                  / sum of IDF over all question terms

    - Terms come from retrieval.tokenize, the same stopwords and stemmer BM25 uses.
      Each distinct term counts once, and Bryan's name is dropped because nearly
      every question uses it.
    - IDF is BM25's (Lucene variant), so a rare term weighs more than a common one.
      A term found in no post at all gets the highest weight and can never match.
    - A term "appears" if it occurs anywhere in any of the posts. Unlike BM25 itself,
      how often it occurs and how long the post is don't matter.

    The result runs from 0 (no shared terms) to 1 (every term appears).
    """

    def __init__(self, posts: dict[Slug, str]) -> None:
        texts = [strip_links(text) for text in posts.values()]
        self.terms_in_post: dict[Slug, set[Term]] = {
            slug: set(terms) for slug, terms in zip(posts, tokenize(texts))}
        self.num_posts = len(posts)
        self.posts_containing: collections.Counter[Term] = collections.Counter()
        for terms in self.terms_in_post.values():
            self.posts_containing.update(terms)

    def idf(self, term: Term) -> float:
        df = self.posts_containing[term]
        return math.log(1 + (self.num_posts - df + 0.5) / (df + 0.5))

    def question_terms(self, question: str) -> set[Term]:
        return set(tokenize(question)[0]) - NAME

    def score(self, question: str, slugs: list[Slug]) -> float:
        """Overlap of `question` with the combined terms of `slugs` (0.0 if `slugs` is empty)."""
        terms = self.question_terms(question)
        if not terms:
            return 0.0
        found = set().union(*(self.terms_in_post[s] for s in slugs))
        matched_weight = sum(self.idf(t) for t in terms & found) #idf for terms that appear in the 
        total_weight = sum(self.idf(t) for t in terms)
        return matched_weight / total_weight

    def explain(self, question: str, slug_groups: dict[str, list[Slug]]) -> list[TermRow]:
        """One row per question term, heaviest first. slug_groups maps a label such
        as "current" to a list of slugs; groups with no slugs are left out."""
        found = {label: set().union(*(self.terms_in_post[s] for s in slugs))
                 for label, slugs in slug_groups.items() if slugs}
        terms = sorted(self.question_terms(question), key=self.idf, reverse=True)
        return [TermRow(t, self.idf(t), {label: t in f for label, f in found.items()})
                for t in terms]


@dataclass(frozen=True)
class Overlap:
    """One question's word overlap with its posts."""
    source: float              # with source_posts
    superseded: float | None   # with superseded_posts; None if the question has none

def word_overlap(qs: list[Question], posts: dict[Slug, str]) -> dict[str, Overlap]:
    """Question id -> its overlap. Every question in `qs` needs source_posts."""
    scorer = WordOverlap(posts)
    out = {}
    for q in qs:
        superseded = (scorer.score(q.question, q.superseded_posts)
                      if q.superseded_posts else None)
        out[q.id] = Overlap(scorer.score(q.question, q.source_posts), superseded)
    return out

def validate(qs: list[Question], posts: dict[Slug, str]) -> list[str]:
    """The checks parsing can't do: unique ids, known slugs, rules that span fields,
    and evidence quotes found in their posts. One message per problem."""
    errs = [f"duplicate id {k}" for k, v in collections.Counter(q.id for q in qs).items() if v > 1]
    for q in qs:
        i = q.id
        for f in POST_FIELDS:
            for s in getattr(q, f):
                if s not in posts: errs.append(f"{i}: {f} has unknown slug {s}")
        if q.type == "unanswerable":
            if q.source_posts: errs.append(f"{i}: unanswerable with source_posts")
            continue
        if not q.source_posts: errs.append(f"{i}: no source_posts")
        if not q.key_facts: errs.append(f"{i}: missing key_facts")
        labeled = [s for f in POST_FIELDS for s in getattr(q, f)]
        errs += [f"{i}: {s} appears in more than one post field" for s, n in collections.Counter(labeled).items() if n > 1]
        for s in q.source_posts:
            if s not in q.evidence: errs.append(f"{i}: no evidence for {s}")
        for s, quote in q.evidence.items():
            if s not in q.source_posts and s not in q.acceptable_posts:
                errs.append(f"{i}: evidence for non-source {s}")
            elif s in posts and not contains(posts[s], quote):
                errs.append(f"{i}: quote not found in {s}: {quote!r}")
    return errs

def print_coverage(qs: list[Question], posts: dict[Slug, str], post_cap: int) -> None:
    print("by type: ", dict(collections.Counter(q.type for q in qs)))
    print("by topic:", dict(collections.Counter(q.topic for q in qs)))
    print("by probe:", dict(collections.Counter(q.probe for q in qs if q.probe)))
    required = collections.Counter(s for q in qs for s in q.source_posts)
    targeted = set(required) | {s for q in qs for s in q.acceptable_posts}
    overused = [(s, n) for s, n in required.most_common() if n > post_cap]
    print(f"posts required by more than {post_cap} questions:", overused or "none")
    print("posts no question targets (distractors only):", sorted(set(posts) - targeted) or "none")
    no_stale = [q.id for q in qs if q.type == "current_state" and not q.superseded_posts]
    print("current_state without a superseded post:", no_stale or "none")

def percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile: the value at position p of the way through the list."""
    return sorted_values[min(len(sorted_values) - 1, int(p * len(sorted_values)))]

def print_overlap(qs: list[Question], posts: dict[Slug, str], list_all: bool) -> None:
    answerable = [q for q in qs if q.source_posts]
    if not answerable:
        return
    overlap = word_overlap(answerable, posts)
    source = {qid: o.source for qid, o in overlap.items()}

    # 1. Distribution across all questions
    values = sorted(source.values())
    print("\nword overlap with source posts (0 = no shared terms, 1 = all, rarity-weighted):")
    print(f"  mean {statistics.mean(values):.2f}  min {values[0]:.2f}  25% {percentile(values, .25):.2f}"
          f"  median {statistics.median(values):.2f}  75% {percentile(values, .75):.2f}  max {values[-1]:.2f}")

    # 2. Average for each question type
    by_type: dict[QuestionType, list[float]] = collections.defaultdict(list)
    for q in answerable:
        by_type[q.type].append(source[q.id])
    print("  mean by type:", {t: round(statistics.mean(v), 2) for t, v in by_type.items()})

    # 3. Current vs outdated post. Where the outdated post overlaps at least as much,
    #    the wording gives BM25 no reason to prefer the current one.
    pairs = [(qid, o.source, o.superseded) for qid, o in overlap.items() if o.superseded is not None]
    if pairs:
        print("  current_state, current vs outdated post (<- means wording gives the current post no edge):")
        for qid, current, outdated in sorted(pairs, key=lambda p: p[1] - p[2]):
            flag = "  <-" if outdated >= current else ""
            print(f"    {qid}: current {current:.2f}  outdated {outdated:.2f}{flag}")

    # 4. Individual questions, lowest overlap first
    text = {q.id: q.question for q in answerable}
    ranked = sorted(source.items(), key=lambda item: item[1])
    if list_all or len(ranked) <= 10:
        print("  per question:")
        sections = [ranked]
    else:
        print("  per question (lowest and highest 5; --overlap lists all):")
        sections = [ranked[:5], ranked[-5:]]
    for i, section in enumerate(sections):
        if i:
            print("    ...")
        for qid, value in section:
            print(f"    {qid:5} {value:.2f}  {text[qid][:80]}")

def print_explanation(q: Question, posts: dict[Slug, str]) -> None:
    """Per-term breakdown of one question's overlap scores."""
    scorer = WordOverlap(posts)
    groups = {"current": q.source_posts, "outdated": q.superseded_posts}
    groups = {label: slugs for label, slugs in groups.items() if slugs}
    print(f"\n{q.id}: {q.question}")
    for label, slugs in groups.items():
        print(f"  {label} posts: {', '.join(slugs)}")
    labels = list(groups)
    print(f"\n  {'term':12} {'posts with it':>13} {'IDF':>6}" + "".join(f" {'in ' + l:>12}" for l in labels))
    for row in scorer.explain(q.question, groups):
        df = scorer.posts_containing[row.term]
        print(f"  {row.term:12} {df:>13} {row.idf:>6.2f}"
              + "".join(f" {'yes' if row.found[l] else '-':>12}" for l in labels))
    for label, slugs in groups.items():
        print(f"  overlap with {label} posts: {scorer.score(q.question, slugs):.2f}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blog", default=str(DEFAULT_BLOG))
    ap.add_argument("--file", default=str(ROOT / "dataset" / "eval-set-v2-shortlist.yaml"))
    ap.add_argument("--post-cap", type=int, default=3, help="warn when a post is required by more questions than this")
    ap.add_argument("--overlap", action="store_true", help="list overlap for every question")
    ap.add_argument("--explain", metavar="ID", help="show the per-term overlap breakdown for one question")
    a = ap.parse_args()
    posts = load_posts(a.blog)
    items = load_items(a.file)
    filled = [item for item in items if not is_blank(item)]
    qs, parse_errs = parse_questions(filled)

    if a.explain:
        match = [q for q in qs if q.id == a.explain]
        if not match:
            problems = [e for e in parse_errs if e.startswith(f"{a.explain}: ")]
            sys.exit(f"no valid question with id {a.explain} in {a.file}"
                     + "".join(f"\n  {e}" for e in problems))
        print_explanation(match[0], posts)
        return

    errs = parse_errs + validate(qs, posts)
    skipped = len(items) - len(filled)
    failed = len(filled) - len(qs)
    print(f"{a.file}: {len(qs)} questions" + (f" ({skipped} blank, skipped)" if skipped else "")
          + (f", {failed} failed to parse (left out below)" if failed else "")
          + f", {len(posts)} posts")
    print_coverage(qs, posts, a.post_cap)
    print_overlap(qs, posts, a.overlap)

    if errs:
        print(f"\nERRORS ({len(errs)}):")
        print("\n".join("  " + e for e in errs))
        sys.exit(1)
    print("\nOK: schema valid, all slugs exist, every evidence quote was found.")

if __name__ == "__main__":
    main()
