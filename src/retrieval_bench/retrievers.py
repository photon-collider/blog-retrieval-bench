"""Retrievers. Each has a `name` and a `rank` method, so evaluate.evaluate() can
run it (see evaluate.Retriever).

Every retriever indexes retrieval.post_text(raw) for each post, so they all see
the same text.
"""

from bm25s import BM25

from retrieval_bench.models import Slug
from retrieval_bench.retrieval import post_text, tokenize


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
        # Document i in the index is self.slugs[i].
        self.slugs = list(posts)
        self.retriever = BM25()
        self.retriever.index(tokenize([post_text(posts[slug]) for slug in self.slugs]))

    def rank(self, questions: list[str]) -> list[list[Slug]]:
        """For each question, every post's slug, highest BM25 score first."""
        output: list[list[Slug]] = []
        for q in tokenize(questions):
            scores = self.retriever.get_scores(q)
            slug_scores = sorted(
                zip(self.slugs, scores), key=lambda x: x[1], reverse=True
            )
            output.append([slug for slug, _ in slug_scores])
        return output
