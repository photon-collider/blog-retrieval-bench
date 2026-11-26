from retrieval_bench.retrievers import BM25Retriever


def mdx(title, body):
    """A minimal post in the blog's MDX format."""
    return f'---\ntitle: "{title}"\ndescription: "A post."\ndate: 2026-01-01\n---\n\n{body}\n'


POSTS = {
    "recipes": mdx("Self-hosting a recipe manager", "I run Mealie to manage my recipes."),
    "nas": mdx("Building a NAS", "The NAS has eight drive bays and runs TrueNAS."),
    "photos": mdx("Film photography", "I shot a roll of Portra 400 in Chicago."),
}


def test_ranks_every_post_for_every_question():
    rankings = BM25Retriever(POSTS).rank(["recipe manager", "drive bays"])
    assert len(rankings) == 2
    for ranking in rankings:
        assert sorted(ranking) == sorted(POSTS)


def test_best_match_ranks_first():
    rankings = BM25Retriever(POSTS).rank(
        ["Which recipe manager does he use?", "How many drive bays?", "What film stock?"])
    assert [r[0] for r in rankings] == ["recipes", "nas", "photos"]


def test_matches_through_the_stemmer():
    # "recipes" in the post and "recipe" in the question share the stem "recip".
    assert BM25Retriever(POSTS).rank(["recipe"])[0][0] == "recipes"


def test_indexes_the_title():
    # "Self-hosting" appears only in the recipes post's frontmatter title.
    assert BM25Retriever(POSTS).rank(["self-hosting"])[0][0] == "recipes"
