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


def get_highest_source_rank(
    ranking: list[Slug], source_posts: list[Slug]
) -> int | None:
    """The rank (1 = top) of the first source post in `ranking`, or None."""
    sources = set(source_posts)
    for rank, post in enumerate(ranking, start=1):
        if post in sources:
            return rank
    return None


def get_recall_at(
    ranking: list[Slug], source_posts: list[Slug], ks: Sequence[int]
) -> dict[int, float]:
    """For each k, the share of `source_posts` in the top k of `ranking`."""
    sources = set(source_posts)
    return {k: len(sources & set(ranking[:k])) / len(sources) for k in ks}


def score(
    retriever: str, q: Question, ranking: list[Slug], ks: Sequence[int] = KS
) -> QuestionResult:
    """Score one ranking against one question's labels.

    Ranks start at 1: ranking[0] is rank 1.
    """
    first_source_rank = get_highest_source_rank(ranking, q.source_posts)
    hit_at = {
        k: first_source_rank is not None and first_source_rank <= k for k in ks
    }
    recall_at = (
        get_recall_at(ranking, q.source_posts, ks) if q.type == "multi_post" else None
    )

    return QuestionResult(
        retriever=retriever,
        question_id=q.id,
        type=q.type,
        ranking=ranking,
        first_source_rank=first_source_rank,
        hit_at=hit_at,
        recall_at=recall_at,
    )


def evaluate(
    retriever: Retriever, questions: list[Question], ks: Sequence[int] = KS
) -> list[QuestionResult]:
    """Rank posts for every answerable question and score each ranking."""
    answerable = [q for q in questions if q.source_posts]
    rankings = retriever.rank([q.question for q in answerable])
    if len(rankings) != len(answerable):
        raise ValueError(
            f"{retriever.name} returned {len(rankings)} rankings "
            f"for {len(answerable)} questions"
        )
    return [
        score(retriever.name, q, ranking, ks)
        for q, ranking in zip(answerable, rankings)
    ]


def _summary(retriever: str, group: str, results: list[QuestionResult]) -> Summary:
    """Average one retriever's results over one group of questions."""
    ks = results[0].hit_at.keys()
    n = len(results)

    hit_at = {k: sum(r.hit_at[k] for r in results) / n for k in ks}

    # Recall only if every result has it (a multi_post group). The "all" row
    # mixes types, so it gets none.
    recalls = [r.recall_at for r in results if r.recall_at is not None]
    recall_at = (
        {k: sum(rec[k] for rec in recalls) / n for k in ks}
        if len(recalls) == n
        else None
    )

    mrr = sum(r.reciprocal_rank for r in results) / n

    return Summary(
        retriever=retriever,
        group=group,
        n=n,
        hit_at=hit_at,
        recall_at=recall_at,
        mrr=mrr,
    )


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
    header = (
        f"{'retriever':16} {'group':14} {'n':>3}"
        + "".join(f" {'Hit@' + str(k):>7}" for k in ks)
        + "".join(f" {'R@' + str(k):>6}" for k in ks)
        + f" {'MRR':>6}"
    )
    print(header)
    for s in summaries:
        recall = (
            f" {s.recall_at[k]:>6.2f}" if s.recall_at else f" {'-':>6}" for k in ks
        )
        print(
            f"{s.retriever:16} {s.group:14} {s.n:>3}"
            + "".join(f" {s.hit_at[k]:>7.2f}" for k in ks)
            + "".join(recall)
            + f" {s.mrr:>6.2f}"
        )
