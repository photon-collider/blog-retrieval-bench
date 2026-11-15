"""Evaluation harness: run a retriever over the eval questions and score it.

Any object with a `name` and a `rank` method is a retriever. The harness asks it
to rank every post for every question, then scores each ranking against the
question's labels:

- Hit@k:    is any source post in the top k?
- MRR:      the mean of 1 / (rank of the first source post).
- Recall@k: multi_post questions only. What share of the source posts is in
            the top k? A question with 3 source posts can't reach 1.0 below
            k = 3. For other types any one source post answers the question,
            so Hit@k already says everything.

Unanswerable questions are left out: with no source posts, there's nothing to
find. Results are summarized per question type and over all questions, since
single_fact questions would otherwise dominate.

    from retrieval_bench.evaluate import evaluate, summarize, print_summaries
    results = evaluate(my_retriever, load_questions())
    print_summaries(summarize(results))
"""
import collections
import statistics
from collections.abc import Sequence
from typing import Protocol

from retrieval_bench.models import Question, QuestionResult, Slug, Summary

KS = (1, 3, 5)


class Retriever(Protocol):
    name: str

    def rank(self, questions: list[str]) -> list[list[Slug]]:
        """For each question, post slugs ordered best first. Ranking every post
        is best; a post left out counts as not retrieved."""
        ...


def score(retriever: str, q: Question, ranking: list[Slug], ks: Sequence[int] = KS) -> QuestionResult:
    """Score one ranking against one question's labels.

    Ranks start at 1: ranking[0] is rank 1.
    """
    # TODO: the rank of the first post in `ranking` that's one of
    #       q.source_posts, or None if none of them appears.
    first_source_rank: int | None = None

    # TODO: for each k in ks, is a source post in the top k?
    hit_at: dict[int, bool] = {}

    # TODO: for multi_post questions only: for each k in ks, the share of
    #       q.source_posts in the top k. For every other type, leave it None.
    recall_at: dict[int, float] | None = None

    return QuestionResult(
        retriever=retriever,
        question_id=q.id,
        type=q.type,
        ranking=ranking,
        first_source_rank=first_source_rank,
        hit_at=hit_at,
        recall_at=recall_at,
    )


def evaluate(retriever: Retriever, questions: list[Question], ks: Sequence[int] = KS) -> list[QuestionResult]:
    """Rank posts for every answerable question and score each ranking."""
    answerable = [q for q in questions if q.source_posts]
    rankings = retriever.rank([q.question for q in answerable])
    if len(rankings) != len(answerable):
        raise ValueError(f"{retriever.name} returned {len(rankings)} rankings "
                         f"for {len(answerable)} questions")
    return [score(retriever.name, q, ranking, ks) for q, ranking in zip(answerable, rankings)]


def _summary(retriever: str, group: str, results: list[QuestionResult]) -> Summary:
    """Average one retriever's results over one group of questions."""
    ks = results[0].hit_at.keys()

    # TODO: for each k, the share of results with a hit in the top k.
    hit_at: dict[int, float] = {}

    # TODO: if every result has a recall_at (a multi_post group), the mean of
    #       their recall at each k. Otherwise None: the "all" row mixes types,
    #       so it gets no recall.
    recall_at: dict[int, float] | None = None

    # TODO: the mean reciprocal rank (QuestionResult.reciprocal_rank).
    mrr: float = 0.0

    return Summary(retriever=retriever, group=group, n=len(results),
                   hit_at=hit_at, recall_at=recall_at, mrr=mrr)


def summarize(results: list[QuestionResult]) -> list[Summary]:
    """Averages per retriever: one row per question type, then one for "all"."""
    by_retriever = collections.defaultdict(list)
    for r in results:
        by_retriever[r.retriever].append(r)
    summaries = []
    for retriever, rs in by_retriever.items():
        by_type = collections.defaultdict(list)
        for r in rs:
            by_type[r.type].append(r)
        summaries += [_summary(retriever, t, group) for t, group in by_type.items()]
        summaries.append(_summary(retriever, "all", rs))
    return summaries


def print_summaries(summaries: list[Summary]) -> None:
    """A plain-text table, one row per summary."""
    if not summaries:
        return
    ks = list(summaries[0].hit_at)
    header = (f"{'retriever':16} {'group':14} {'n':>3}"
              + "".join(f" {'Hit@' + str(k):>7}" for k in ks)
              + "".join(f" {'R@' + str(k):>6}" for k in ks) + f" {'MRR':>6}")
    print(header)
    for s in summaries:
        recall = (f" {s.recall_at[k]:>6.2f}" if s.recall_at else f" {'-':>6}" for k in ks)
        print(f"{s.retriever:16} {s.group:14} {s.n:>3}"
              + "".join(f" {s.hit_at[k]:>7.2f}" for k in ks)
              + "".join(recall) + f" {s.mrr:>6.2f}")
