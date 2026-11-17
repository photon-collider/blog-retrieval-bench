"""Question-by-question comparisons for the analysis notebook.

Everything here works on saved results (models.load_results), except
term_contributions, which needs a BM25Retriever. Plain Python in, plain Python
out; the notebook turns the output into tables.
"""
from collections import Counter
from collections.abc import Sequence
from math import comb

from retrieval_bench.evaluate import KS
from retrieval_bench.models import Question, QuestionResult, Slug
from retrieval_bench.retrieval import tokenize
from retrieval_bench.retrievers import BM25Retriever


def by_retriever(results: list[QuestionResult]) -> dict[str, dict[str, QuestionResult]]:
    """retriever name -> question id -> result, in the order the results came."""
    table: dict[str, dict[str, QuestionResult]] = {}
    for r in results:
        table.setdefault(r.retriever, {})[r.question_id] = r
    return table


def random_baseline(questions: list[Question], n_posts: int, ks: Sequence[int] = KS) -> dict:
    """Expected Hit@k and MRR of a uniformly random ranking of `n_posts` posts,
    averaged over the answerable questions. A question with m source posts:

    - P(no source in the top k) = C(n - m, k) / C(n, k), so Hit@k is 1 minus that.
    - P(the first source is at rank r) = C(n - r, m - 1) / C(n, m): the other
      m - 1 sources all fall below rank r. MRR sums that times 1 / r.
    """
    answerable = [q for q in questions if q.source_posts]
    n = n_posts
    hit = {k: 0.0 for k in ks}
    mrr = 0.0
    for q in answerable:
        m = len(q.source_posts)
        for k in ks:
            hit[k] += 1 - comb(n - m, k) / comb(n, k)
        mrr += sum(comb(n - r, m - 1) / comb(n, m) / r for r in range(1, n - m + 2))
    return {"hit_at": {k: h / len(answerable) for k, h in hit.items()},
            "mrr": mrr / len(answerable)}


def agreement(results: list[QuestionResult], a: str, b: str, k: int = 1) -> dict[tuple[bool, bool], list[str]]:
    """The 2x2 table of retriever `a` against retriever `b` at Hit@k:
    (a hit?, b hit?) -> question ids, in eval-set order."""
    table = by_retriever(results)
    cells: dict[tuple[bool, bool], list[str]] = {
        (True, True): [], (True, False): [], (False, True): [], (False, False): []}
    for qid, ra in table[a].items():
        cells[(ra.hit_at[k], table[b][qid].hit_at[k])].append(qid)
    return cells


def paraphrase_flips(original: list[QuestionResult], paraphrase: list[QuestionResult],
                     retriever: str, k: int = 1) -> tuple[list[str], list[str]]:
    """Question ids whose Hit@k changes between the wordings for one retriever:
    (lost: hit on the original only, gained: hit on the paraphrase only)."""
    before, after = by_retriever(original)[retriever], by_retriever(paraphrase)[retriever]
    lost = [qid for qid, r in before.items() if r.hit_at[k] and not after[qid].hit_at[k]]
    gained = [qid for qid, r in before.items() if not r.hit_at[k] and after[qid].hit_at[k]]
    return lost, gained


def term_contributions(bm25: BM25Retriever, question: str, slugs: list[Slug]) -> list[dict]:
    """How much each query term adds to each post's BM25 score.

    BM25 is a sum over query terms, and bm25s counts a repeated term once per
    occurrence, so a term's part is its count times its score on its own. The
    parts add up to the post's full score. `posts_with_term` is the term's
    document frequency: the fewer posts, the higher its IDF.

    One dict per distinct term, in query order: term, count, posts_with_term,
    then one entry per slug in `slugs`.
    """
    counts = Counter(tokenize(question)[0])
    rows = []
    for term, count in counts.items():
        scores = bm25.retriever.get_scores([term])  # zeros for a term no post has
        rows.append({"term": term, "count": count, "posts_with_term": int((scores > 0).sum()),
                     **{slug: count * float(scores[bm25.slugs.index(slug)]) for slug in slugs}})
    return rows
