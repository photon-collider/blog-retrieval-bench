import pytest

from retrieval_bench.models import Question, paraphrased


def question(id, source_posts, paraphrase=None):
    return Question(id=id, type="single_fact", topic="personal", question=f"original {id}",
                    source_posts=source_posts, reference_answer="x", paraphrase=paraphrase)


def test_paraphrased_swaps_in_the_paraphrase():
    q = question("a", ["post"], paraphrase="reworded a")
    [p] = paraphrased([q])
    assert p.question == "reworded a"
    assert p.source_posts == q.source_posts
    assert q.question == "original a"  # the original is unchanged


def test_paraphrased_requires_one_for_every_answerable_question():
    with pytest.raises(ValueError, match="'b'"):
        paraphrased([question("a", ["post"], paraphrase="reworded a"), question("b", ["post"])])


def test_paraphrased_leaves_unanswerable_questions_alone():
    q = question("u", [])
    assert paraphrased([q]) == [q]
