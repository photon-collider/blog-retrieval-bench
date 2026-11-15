import pytest

from retrieval_bench.evaluate import evaluate, score, summarize
from retrieval_bench.models import Question, load_questions


def question(id="q1", type="single_fact", sources=("a",)):
    return Question(id=id, type=type, topic="personal", question=f"question {id}?",
                    source_posts=list(sources), reference_answer="answer")


class FixedRetriever:
    """Returns the same ranking for every question."""
    def __init__(self, ranking, name="fixed"):
        self.name, self.ranking = name, ranking

    def rank(self, questions):
        return [self.ranking for _ in questions]


def test_score_by_hand():
    # Sources a, b and c. The ranking puts c 2nd, a 3rd and b 5th.
    r = score("fixed", question(sources=["a", "b", "c"]), ["x", "c", "a", "y", "b"])
    assert r.first_source_rank == 2            # c comes first, and any source counts
    assert r.reciprocal_rank == 0.5
    assert r.hit_at == {1: False, 3: True, 5: True}
    assert r.recall_at is None                 # not a multi_post question


def test_score_multi_post_recall_by_hand():
    # Same ranking, but now all three posts are needed: recall counts them.
    r = score("fixed", question(type="multi_post", sources=["a", "b", "c"]), ["x", "c", "a", "y", "b"])
    assert r.first_source_rank == 2
    assert r.recall_at == {1: 0.0, 3: 2 / 3, 5: 1.0}


def test_score_nothing_relevant_ranked():
    r = score("fixed", question(sources=["a"]), ["x", "y"])
    assert r.first_source_rank is None
    assert r.reciprocal_rank == 0.0
    assert not any(r.hit_at.values())


def test_evaluate_skips_unanswerable():
    qs = [question("q1"), question("un1", type="unanswerable", sources=())]
    results = evaluate(FixedRetriever(["a"]), qs)
    assert [r.question_id for r in results] == ["q1"]


def test_evaluate_rejects_wrong_number_of_rankings():
    class Broken(FixedRetriever):
        def rank(self, questions):
            return []
    with pytest.raises(ValueError):
        evaluate(Broken(["a"]), [question()])


def test_summarize_groups_by_type():
    qs = [question("s1"), question("s2"), question("m1", type="multi_post", sources=["a", "b"])]
    summaries = {s.group: s for s in summarize(evaluate(FixedRetriever(["a", "x", "b"]), qs))}
    assert {g: s.n for g, s in summaries.items()} == {"single_fact": 2, "multi_post": 1, "all": 3}
    assert summaries["multi_post"].recall_at == {1: 0.5, 3: 1.0, 5: 1.0}
    assert summaries["single_fact"].recall_at is None
    assert summaries["all"].recall_at is None  # mixes types
    assert summaries["all"].mrr == 1.0


def test_oracle_on_real_eval_set_scores_perfectly():
    """A retriever that ranks each question's source posts first, on the real
    eval set: every score should be at its ceiling."""
    qs = load_questions()
    by_text = {q.question: q for q in qs}

    class Oracle:
        name = "oracle"
        def rank(self, questions):
            return [by_text[t].source_posts + ["unrelated"] for t in questions]

    results = evaluate(Oracle(), qs)
    assert len(results) == len([q for q in qs if q.source_posts])
    for r, q in zip(results, (q for q in qs if q.source_posts)):
        assert r.first_source_rank == 1
        if q.type == "multi_post":
            assert r.recall_at[5] == min(5, len(q.source_posts)) / len(q.source_posts)


def test_summary_averages_hits_and_misses():
    # s1's source is ranked first; s2's source isn't ranked at all.
    qs = [question("s1", sources=["a"]), question("s2", sources=["z"])]
    summary = {s.group: s for s in summarize(evaluate(FixedRetriever(["a", "b"]), qs))}["all"]
    assert summary.hit_at == {1: 0.5, 3: 0.5, 5: 0.5}
    assert summary.recall_at is None
    assert summary.mrr == 0.5  # (1/1 + 0) / 2
