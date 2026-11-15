import pytest

from retrieval_bench.evaluate import evaluate, score, summarize
from retrieval_bench.models import Question, load_questions


def question(id="q1", type="single_fact", sources=("a",), acceptable=()):
    return Question(id=id, type=type, topic="personal", question=f"question {id}?",
                    source_posts=list(sources), acceptable_posts=list(acceptable),
                    reference_answer="answer")


class FixedRetriever:
    """Returns the same ranking for every question."""
    def __init__(self, ranking, name="fixed"):
        self.name, self.ranking = name, ranking

    def rank(self, questions):
        return [self.ranking for _ in questions]


def test_score_by_hand():
    # Sources a and b, acceptable c. Ranking puts c 2nd, a 3rd, b 5th.
    r = score("fixed", question(sources=["a", "b"], acceptable=["c"]), ["x", "c", "a", "y", "b"])
    assert r.first_relevant_rank == 2          # c is acceptable, so it counts
    assert r.reciprocal_rank == 0.5
    assert r.hit_at == {1: False, 3: True, 5: True}
    assert r.recall_at == {1: 0.0, 3: 0.5, 5: 1.0}  # recall counts sources only, not c


def test_score_nothing_relevant_ranked():
    r = score("fixed", question(sources=["a"]), ["x", "y"])
    assert r.first_relevant_rank is None
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
        assert r.first_relevant_rank == 1
        assert r.recall_at[5] == min(5, len(q.source_posts)) / len(q.source_posts)
