"""Retrievers. Each has a `name` and a `rank` method, so evaluate.evaluate() can
run it (see evaluate.Retriever).

Every retriever indexes retrieval.post_text(raw) for each post, so they all see
the same text.
"""
from retrieval_bench.models import Slug
from retrieval_bench.retrieval import post_text, tokenize  # noqa: F401 (you'll use these)


class BM25Retriever:
    """BM25 over whole posts, using bm25s with its defaults (Lucene variant,
    k1 = 1.5, b = 0.75) and retrieval.tokenize for both posts and questions.

        retriever = BM25Retriever(load_posts())
        retriever.rank(["What machine runs Bryan's recipe manager?"])
        # -> [["my-homelab-setup", "building-a-better-nas-for-my-homelab", ...]]
    """

    name = "bm25"

    def __init__(self, posts: dict[Slug, str]) -> None:
        """Build the index. `posts` maps slug -> raw MDX, as load_posts() returns."""
        # TODO: keep the slugs in a fixed order, so a document's position in the
        #       index maps back to its slug.
        # TODO: turn each post into post_text, tokenize it, and index it with bm25s.
        raise NotImplementedError

    def rank(self, questions: list[str]) -> list[list[Slug]]:
        """For each question, every post's slug, highest BM25 score first."""
        # TODO: tokenize each question the same way as the posts, score every
        #       post, and sort the slugs by score.
        raise NotImplementedError
