"""Validate an eval set against the blog posts and report coverage.

Checks the schema (the Question model below), post slugs, and that each required
source post has an evidence quote found in that post. Quotes are looked up in
retrieval.post_text, the text the retrievers index, so an answer that exists only
in dropped markup is caught. A found quote shows the post has the text, not that
the reference answer is right.

Deliberately reports nothing about how a retriever would do: this runs while
questions are being written, and a number shown then would shape the wording.
Retrieval analysis belongs in the notebook, after the questions are frozen.

Items with a blank `question` are skipped, so the v2 workbook (eval-set-v2.yaml)
can be checked too.

Usage: uv run check-eval-set [--file dataset/eval-set-v2-shortlist.yaml] [--blog PATH]
                             [--post-cap 3]
"""
import argparse
import collections
import functools
import re
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from retrieval_bench.retrieval import DEFAULT_BLOG, load_posts, post_text, strip_links

Slug = str  # a post's filename without .mdx, e.g. "my-homelab"

QuestionType = Literal["single_fact", "multi_post", "current_state", "opinion", "unanswerable"]
Topic = Literal["ai-llm", "ml-nlp", "backend", "web-design", "homelab", "self-hosting",
                "computing", "personal", "writing-productivity", "cross-topic"]
Probe = Literal["vocab_mismatch", "lexical_collision", "sibling_confusion", "near_miss",
                "caption_only", "alt_text_only", "table_only", "code_only", "callout_only",
                "buried_detail", "short_post", "long_post", "recency", "false_premise",
                "aggregation", "multi_hop"]

POST_FIELDS = ("source_posts", "acceptable_posts", "superseded_posts", "related_posts")
ROOT = Path(__file__).resolve().parents[2]  # project root (src/retrieval_bench/ is two levels down)


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


PUNCTUATION = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})

def norm(s: str) -> str:
    s = s.translate(PUNCTUATION)
    s = s.replace("\\", "").replace("`", "").replace("**", "")
    return re.sub(r"\s+", " ", s).lower().strip()

@functools.cache
def normalized_forms(raw: str) -> tuple[str, str]:
    """A post normalized with and without its links. Cached, since one post is
    checked against several quotes."""
    return norm(raw), norm(strip_links(raw))

def contains(raw: str, quote: str) -> bool:
    q = norm(quote)
    return any(q in form for form in normalized_forms(raw))

def load_items(path: str | Path) -> list[dict[str, Any]]:
    """Every item in the file as a plain dict, blank `question` or not."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["questions"]

def is_blank(item: dict[str, Any]) -> bool:
    """A blank `question` marks a workbook item that wasn't chosen."""
    return not (item.get("question") or "").strip()

def parse_questions(items: list[dict[str, Any]]) -> tuple[list[Question], dict[str, list[str]]]:
    """Parse each item into a Question. Returns the ones that parsed, and for the
    ones that didn't, id -> one "field: problem" message per problem."""
    questions, errors = [], collections.defaultdict(list)
    for item in items:
        try:
            questions.append(Question.model_validate(item))
        except ValidationError as e:
            qid = str(item.get("id", "?"))
            for err in e.errors():
                field = ".".join(str(part) for part in err["loc"])
                errors[qid].append(f"{field}: {err['msg']}")
    return questions, dict(errors)


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

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blog", default=str(DEFAULT_BLOG))
    ap.add_argument("--file", default=str(ROOT / "dataset" / "eval-set-v2-shortlist.yaml"))
    ap.add_argument("--post-cap", type=int, default=3, help="warn when a post is required by more questions than this")
    a = ap.parse_args()
    posts = {slug: post_text(raw) for slug, raw in load_posts(a.blog).items()}
    items = load_items(a.file)
    filled = [item for item in items if not is_blank(item)]
    qs, parse_errs = parse_questions(filled)

    errs = [f"{qid}: {e}" for qid, msgs in parse_errs.items() for e in msgs] + validate(qs, posts)
    skipped = len(items) - len(filled)
    failed = len(filled) - len(qs)
    print(f"{a.file}: {len(qs)} questions" + (f" ({skipped} blank, skipped)" if skipped else "")
          + (f", {failed} failed to parse (left out below)" if failed else "")
          + f", {len(posts)} posts")
    print_coverage(qs, posts, a.post_cap)

    if errs:
        print(f"\nERRORS ({len(errs)}):")
        print("\n".join("  " + e for e in errs))
        sys.exit(1)
    print("\nOK: schema valid, all slugs exist, every evidence quote was found in the indexed post text.")

if __name__ == "__main__":
    main()
