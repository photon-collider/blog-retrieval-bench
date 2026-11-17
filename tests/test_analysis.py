import pytest

from retrieval_bench.analysis import agreement, paraphrase_flips, random_baseline, term_contributions
from retrieval_bench.evaluate import score
from retrieval_bench.models import Question, load_results, save_results
from retrieval_bench.retrieval import tokenize
from retrieval_bench.retrievers import BM25Retriever


def mdx(title, body):
    return f'---\ntitle: "{title}"\ndescription: "A post."\ndate: 2026-01-01\n---\n\n{body}\n'


POSTS = {
    "recipes": mdx("Self-hosting a recipe manager", "I run Mealie to manage my recipes."),
    "nas": mdx("Building a NAS", "The NAS has eight drive bays and runs TrueNAS."),
    "photos": mdx("Film photography", "I shot a roll of Portra 400 in Chicago."),
}


def question(id, sources=("a",)):
    return Question(id=id, type="single_fact", topic="personal", question=f"question {id}?",
                    source_posts=list(sources), reference_answer="answer")


Q1, Q2, Q3 = question("q1"), question("q2"), question("q3")


def results(name, rankings):
    """One retriever's results for q1..q3, each with source post "a"."""
    return [score(name, q, r) for q, r in zip([Q1, Q2, Q3], rankings)]


def test_random_baseline_one_source_by_hand():
    # One source among 4 posts: Hit@1 = 1/4, Hit@3 = 3/4, MRR = (1 + 1/2 + 1/3 + 1/4) / 4.
    b = random_baseline([question("q", ["a"])], n_posts=4, ks=(1, 3))
    assert b["hit_at"] == pytest.approx({1: 0.25, 3: 0.75})
    assert b["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 3 + 1 / 4) / 4)


def test_random_baseline_two_sources_by_hand():
    # Sources a and b among 3 posts. The first source is at rank 1 unless the
    # one non-source is first (P = 1/3), and then it's at rank 2.
    b = random_baseline([question("q", ["a", "b"])], n_posts=3, ks=(1,))
    assert b["hit_at"][1] == pytest.approx(2 / 3)
    assert b["mrr"] == pytest.approx(2 / 3 * 1 + 1 / 3 * 1 / 2)


def test_agreement_cells():
    x = results("x", [["a", "b"], ["a", "b"], ["b", "a"]])  # hits q1, q2
    y = results("y", [["a", "b"], ["b", "a"], ["a", "b"]])  # hits q1, q3
    cells = agreement(x + y, "x", "y", k=1)
    assert cells == {(True, True): ["q1"], (True, False): ["q2"],
                     (False, True): ["q3"], (False, False): []}


def test_paraphrase_flips():
    before = results("x", [["a", "b"], ["b", "a"], ["a", "b"]])  # hits q1, q3
    after = results("x", [["b", "a"], ["a", "b"], ["a", "b"]])   # hits q2, q3
    assert paraphrase_flips(before, after, "x", k=1) == (["q1"], ["q2"])


def test_term_contributions_add_up_to_the_score():
    bm25 = BM25Retriever(POSTS)
    q = "Which recipe manager runs on his NAS? recipe"  # "recip" twice
    rows = term_contributions(bm25, q, list(POSTS))
    full = dict(zip(bm25.slugs, bm25.retriever.get_scores(tokenize(q)[0])))
    for slug in POSTS:
        assert sum(r[slug] for r in rows) == pytest.approx(full[slug])
    recip = next(r for r in rows if r["term"] == "recip")
    assert recip["count"] == 2 and recip["posts_with_term"] == 1


def test_results_round_trip(tmp_path):
    rs = results("x", [["a", "b"], ["b", "a"], ["a", "b"]])
    save_results(tmp_path / "r.json", rs)
    assert load_results(tmp_path / "r.json") == rs
