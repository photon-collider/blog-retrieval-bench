"""Validate an eval set against the blog posts and report coverage.

Checks the schema (models.Question), post slugs, and that each required
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

from retrieval_bench.models import (
    EVAL_SET,
    POST_FIELDS,
    Question,
    Slug,
    is_blank,
    load_items,
    parse_questions,
)
from retrieval_bench.retrieval import DEFAULT_BLOG, load_posts, post_text, strip_links

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
    ap.add_argument("--file", default=str(EVAL_SET))
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
