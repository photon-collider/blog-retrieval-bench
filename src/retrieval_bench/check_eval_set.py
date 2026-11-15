#!/usr/bin/env -S uv run --quiet --with pyyaml python
"""Validate eval-set.yaml against the blog posts.

Checks the schema, post slugs, and that each required source post has an evidence
quote found in that post. A found quote shows the post has the text, not that the
reference answer is right. Also reports how much each phrasing's wording overlaps
with its source posts, so a paraphrase that repeats the post's rare words stands out.

Usage: ./check_eval_set.py [--blog PATH]
"""
import argparse, collections, math, re, statistics, sys
from pathlib import Path
import yaml

TYPES = {"single_fact", "multi_post", "current_state", "opinion", "unanswerable"}
TOPICS = {"ai-llm", "ml-nlp", "backend", "web-design", "homelab", "self-hosting",
          "computing", "personal", "writing-productivity", "cross-topic"}
PROBES = {"vocab_mismatch", "lexical_collision", "sibling_confusion", "near_miss",
          "caption_only", "alt_text_only", "table_only", "code_only", "callout_only",
          "buried_detail", "short_post", "long_post", "recency", "false_premise",
          "aggregation", "multi_hop"}
POST_FIELDS = ("source_posts", "acceptable_posts", "superseded_posts", "related_posts")
STOP = set("""a about after all also am an and any are as at be been before but by can could
did do does doing down for from get got had has have he her him his how i if in into is it
its just me more most much my no not now of off on one or other our out over own same she
should so some still such than that the their them then there these they this those to too
under up very was we were what when where which while who why will with would you your
s t re ve ll d m""".split())

NAME = {"bryan", "bryan's"}

def norm(s):
    s = s.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"}))
    s = s.replace("\\", "").replace("`", "").replace("**", "")
    return re.sub(r"\s+", " ", s).lower().strip()

def strip_links(s):
    return re.sub(r"\[([^\]]*)\]\(\s*<?[^)\s>]*>?\s*\)", r"\1", s)

def contains(raw, quote):
    q = norm(quote)
    return q in norm(raw) or q in norm(strip_links(raw))

def toks(s):
    return {w for w in re.findall(r"[a-z0-9][a-z0-9'+.-]*[a-z0-9]|[a-z0-9]", s.lower()) if w not in STOP}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blog", default=str(Path.home() / "Projects/personal-blog-astro/src/content/blog"))
    ap.add_argument("--file", default="eval-set.yaml")
    a = ap.parse_args()
    posts = {p.stem: p.read_text(encoding="utf-8") for p in Path(a.blog).glob("*.mdx")}
    qs = yaml.safe_load(open(a.file, encoding="utf-8"))["questions"]
    errs, ids = [], collections.Counter(q["id"] for q in qs)
    errs += [f"duplicate id {k}" for k, v in ids.items() if v > 1]
    overlap = []
    # Weight words by rarity across posts, since rare words drive BM25 and common ones barely matter.
    df = collections.Counter(w for p in posts.values() for w in toks(strip_links(p)))
    idf = lambda w: math.log(1 + (len(posts) - df[w] + 0.5) / (df[w] + 0.5))
    for q in qs:
        i = q["id"]
        for f in ("type", "topic", "question", "paraphrase", "reference_answer"):
            if not q.get(f):
                errs.append(f"{i}: missing {f}")
        if q.get("type") not in TYPES: errs.append(f"{i}: bad type")
        if q.get("topic") not in TOPICS: errs.append(f"{i}: bad topic {q.get('topic')}")
        if q.get("probe") and q["probe"] not in PROBES: errs.append(f"{i}: bad probe {q['probe']}")
        for f in POST_FIELDS:
            for s in q.get(f) or []:
                if s not in posts: errs.append(f"{i}: {f} has unknown slug {s}")
        src = q.get("source_posts") or []
        if q["type"] == "unanswerable":
            if src: errs.append(f"{i}: unanswerable with source_posts")
            continue
        if not src: errs.append(f"{i}: no source_posts")
        ev = q.get("evidence") or {}
        for s in src:
            if s not in ev: errs.append(f"{i}: no evidence for {s}")
        for s, quote in ev.items():
            if s not in src and s not in (q.get("acceptable_posts") or []):
                errs.append(f"{i}: evidence for non-source {s}")
            elif s in posts and not contains(posts[s], str(quote)):
                errs.append(f"{i}: quote not found in {s}: {quote!r}")
        vocab = set().union(*(toks(strip_links(posts[s])) for s in src if s in posts)) if src else set()
        qt, pt = toks(q["question"]) - NAME, toks(q["paraphrase"]) - NAME
        ov = lambda t: sum(idf(w) for w in t & vocab) / max(sum(idf(w) for w in t), 1e-9)
        shared = len(qt & pt) / max(len(qt | pt), 1)
        overlap.append((i, ov(qt), ov(pt), shared))
    # Coverage
    req = collections.Counter(s for q in qs for s in (q.get("source_posts") or []))
    uncovered = sorted(set(posts) - set(req))
    print(f"questions: {len(qs)}  posts: {len(posts)}  posts with no question: {uncovered or 'none'}")
    print("by type: ", dict(collections.Counter(q['type'] for q in qs)))
    print("by topic:", dict(collections.Counter(q['topic'] for q in qs)))
    print("by probe:", dict(collections.Counter(q['probe'] for q in qs if q.get('probe'))))
    print("most-used source posts:", req.most_common(5))
    print(f"\nrarity-weighted word overlap with source posts (mean): question {statistics.mean(o[1] for o in overlap):.2f}"
          f"  paraphrase {statistics.mean(o[2] for o in overlap):.2f}")
    weak = [o for o in overlap if o[2] >= o[1] - 0.05 or o[3] >= 0.5]
    print(f"weak paraphrases (overlap within 0.05 of the question, or >=50% shared words): {len(weak)}")
    for i, q_, p_, sh in sorted(weak, key=lambda o: -o[3]):
        print(f"  {i}: question {q_:.2f}  paraphrase {p_:.2f}  shared {sh:.2f}")
    if errs:
        print(f"\nERRORS ({len(errs)}):")
        print("\n".join("  " + e for e in errs))
        sys.exit(1)
    print("\nOK: schema valid, all slugs exist, every evidence quote was found.")

if __name__ == "__main__":
    main()
