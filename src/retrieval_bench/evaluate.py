"""Evaluation harness: run a retriever over the eval questions and score it.

Any object with a `name` and a `rank` method is a retriever. The harness asks it
to rank every post for every question, then scores each ranking against the
question's labels:

- Hit@k:    is a source or acceptable post in the top k?
- Recall@k: what share of the source posts is in the top k? A multi_post
            question with 3 source posts can't reach 1.0 below k = 3.
- MRR:      the mean of 1 / (rank of the first source or acceptable post).

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
    """Score one ranking against one question's labels."""
    relevant = set(q.source_posts) | set(q.acceptable_posts)
    sources = set(q.source_posts)
    first = next((i for i, slug in enumerate(ranking, start=1) if slug in relevant), None)
    return QuestionResult(
        retriever=retriever,
        question_id=q.id,
        type=q.type,
        ranking=ranking,
        first_relevant_rank=first,
        hit_at={k: first is not None and first <= k for k in ks},
        recall_at={k: len(sources & set(ranking[:k])) / len(sources) for k in ks},
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
    ks = results[0].hit_at.keys()
    return Summary(
        retriever=retriever,
        group=group,
        n=len(results),
        hit_at={k: statistics.mean(r.hit_at[k] for r in results) for k in ks},
        recall_at={k: statistics.mean(r.recall_at[k] for r in results) for k in ks},
        mrr=statistics.mean(r.reciprocal_rank for r in results),
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
    header = (f"{'retriever':16} {'group':14} {'n':>3}"
              + "".join(f" {'Hit@' + str(k):>7}" for k in ks)
              + "".join(f" {'R@' + str(k):>6}" for k in ks) + f" {'MRR':>6}")
    print(header)
    for s in summaries:
        print(f"{s.retriever:16} {s.group:14} {s.n:>3}"
              + "".join(f" {s.hit_at[k]:>7.2f}" for k in ks)
              + "".join(f" {s.recall_at[k]:>6.2f}" for k in ks) + f" {s.mrr:>6.2f}")
