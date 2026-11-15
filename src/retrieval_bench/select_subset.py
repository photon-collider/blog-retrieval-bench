"""Pick a stratified eval subset from eval-set.yaml. Never looks at retriever results.

v1 only: the v2 eval set is hand-picked and doesn't use this sampler.

Strata:
  type   fixed shares, so each question type has enough questions to read
  topic  proportional to each topic's share of posts, not its share of questions
  post   no post can be a required source for more than --post-cap questions
  stale  most current_state picks must have an outdated post that competes with
         the right one (superseded_posts); the rest only test recent facts

Multi-post questions with more required posts than --max-sources are held out,
because top-k retrieval can't reach full recall on them at the k values planned.
A fixed seed makes the pick reproducible. Choose --seed and --n once, before the
first retrieval run, and don't change them after seeing results.

Usage: uv run select-subset [--n 40] [--seed 20261105] [--post-cap 3]
"""
import argparse, collections, random
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]  # project root (src/retrieval_bench/ is two levels down)

TYPE_SHARE = {"single_fact": .35, "current_state": .175, "multi_post": .175, "opinion": .15, "unanswerable": .15}
POST_TOPIC = {
    "ai-llm": 5, "ml-nlp": 2, "backend": 5, "web-design": 10, "homelab": 3,
    "self-hosting": 3, "computing": 2, "personal": 8, "writing-productivity": 5,
}  # posts per topic, totals 43
CROSS_SHARE = .05  # cross-topic questions don't belong to one topic's posts
STALE_SHARE = .7   # minimum share of current_state picks with a stale competitor

def largest_remainder(shares, n):
    total = sum(shares.values())
    raw = {k: v / total * n for k, v in shares.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[: n - sum(out.values())]:
        out[k] += 1
    return out

def score(sel, topic_targets, cap, cell_cap):
    topics = collections.Counter(q["topic"] for q in sel)
    posts = collections.Counter(s for q in sel for s in (q.get("source_posts") or []))
    cells = collections.Counter((q["type"], q["topic"]) for q in sel)
    probes = {q["probe"] for q in sel if q.get("probe")}
    cost = 10 * sum((topics[t] - v) ** 2 for t, v in topic_targets.items())
    cost += 100 * sum(max(0, c - cap) ** 2 for c in posts.values())
    cost += 5 * sum(max(0, c - cell_cap) ** 2 for c in cells.values())
    current = [q for q in sel if q["type"] == "current_state"]
    stale = sum(bool(q.get("superseded_posts")) for q in current)
    cost += 100 * max(0, round(STALE_SHARE * len(current)) - stale) ** 2
    cost -= 2 * len(posts) + len(probes)
    return cost

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261105)
    ap.add_argument("--post-cap", type=int, default=3)
    ap.add_argument("--cell-cap", type=int, default=3)
    ap.add_argument("--max-sources", type=int, default=5)
    ap.add_argument("--restarts", type=int, default=60)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", default=str(ROOT / "dataset" / "eval-subset.yaml"))
    a = ap.parse_args()

    data = yaml.safe_load((ROOT / "dataset" / "eval-set.yaml").read_text(encoding="utf-8"))
    pool, held = [], []
    for q in data["questions"]:
        (held if len(q.get("source_posts") or []) > a.max_sources else pool).append(q)
    by_type = collections.defaultdict(list)
    for q in pool:
        by_type[q["type"]].append(q)

    type_quota = largest_remainder(TYPE_SHARE, a.n)
    cross = round(a.n * CROSS_SHARE)
    topic_targets = largest_remainder(POST_TOPIC, a.n - cross)
    topic_targets["cross-topic"] = cross

    rng = random.Random(a.seed)
    best, best_cost = None, float("inf")
    for _ in range(a.restarts):
        sel = {t: rng.sample(sorted(qs, key=lambda q: q["id"]), type_quota[t]) for t, qs in sorted(by_type.items())}
        flat = lambda: [q for t in sorted(sel) for q in sel[t]]
        cost = score(flat(), topic_targets, a.post_cap, a.cell_cap)
        for _ in range(a.steps):
            t = rng.choice(sorted(sel))
            i = rng.randrange(len(sel[t]))
            candidate = rng.choice(by_type[t])
            if candidate in sel[t]:
                continue
            old, sel[t][i] = sel[t][i], candidate
            new_cost = score(flat(), topic_targets, a.post_cap, a.cell_cap)
            if new_cost <= cost:
                cost = new_cost
            else:
                sel[t][i] = old
        if cost < best_cost:
            best, best_cost = flat(), cost

    best.sort(key=lambda q: (list(TYPE_SHARE).index(q["type"]), q["id"]))
    header = {
        "as_of": data["as_of"],
        "sampling": {"n": a.n, "seed": a.seed, "post_cap": a.post_cap, "cell_cap": a.cell_cap,
                     "max_sources": a.max_sources, "source": "eval-set.yaml",
                     "held_out": [q["id"] for q in held]},
    }
    Path(a.out).write_text(
        "# Generated by select_subset.py. Don't hand-edit; change the parameters and regenerate.\n"
        + yaml.safe_dump({**header, "questions": best}, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8")

    topics = sorted({q["topic"] for q in best} | set(topic_targets))
    types = list(TYPE_SHARE)
    print(f"selected {len(best)} from a pool of {len(pool)} (held out: {[q['id'] for q in held]}) -> {a.out}\n")
    print(f"{'':22}" + "".join(f"{t[:7]:>9}" for t in types) + f"{'total':>7}{'target':>7}")
    cells = collections.Counter((q["type"], q["topic"]) for q in best)
    for tp in topics:
        row = [cells[(t, tp)] for t in types]
        print(f"{tp:22}" + "".join(f"{c or '.':>9}" for c in row) + f"{sum(row):>7}{topic_targets.get(tp, 0):>7}")
    print(f"{'total':22}" + "".join(f"{sum(cells[(t, tp)] for tp in topics):>9}" for t in types) + f"{len(best):>7}")
    posts = collections.Counter(s for q in best for s in (q.get("source_posts") or []))
    print(f"\nposts covered as required sources: {len(posts)}/43; most used: {posts.most_common(5)}")
    current = [q for q in best if q["type"] == "current_state"]
    print(f"current_state with a stale competitor: {sum(bool(q.get('superseded_posts')) for q in current)}/{len(current)}")
    print("posts not required by any pick:", sorted({p.stem for p in Path.home().joinpath("Projects/personal-blog-astro/src/content/blog").glob("*.mdx")} - set(posts)))
    print("probes:", dict(collections.Counter(q["probe"] for q in best if q.get("probe"))))
    print("from round 3:", sum(q["id"][2:].isdigit() and (
        (q["id"][:2] == "sf" and int(q["id"][2:]) > 35) or (q["id"][:2] == "cs" and int(q["id"][2:]) > 9) or
        (q["id"][:2] == "mp" and int(q["id"][2:]) > 10) or (q["id"][:2] == "op" and int(q["id"][2:]) > 8) or
        (q["id"][:2] == "un" and int(q["id"][2:]) > 8)) for q in best), "of", len(best))
    print("\nids:", " ".join(q["id"] for q in best))

if __name__ == "__main__":
    main()
