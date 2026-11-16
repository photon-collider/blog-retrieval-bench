"""EmbeddingRetriever with a fake model (no download), and RRFRetriever with
fixed rankings."""
import numpy as np
import pytest

from retrieval_bench.retrievers import EmbeddingRetriever, RRFRetriever


def mdx(body):
    """A minimal post in the blog's MDX format."""
    return f'---\ntitle: "A post"\ndate: 2026-01-01\n---\n\n{body}\n'


POSTS = {
    "recipes": mdx("I cook dinner from my recipe manager."),
    "nas": mdx("My storage server has eight drive bays."),
    "photos": mdx("I shot a roll of film in Chicago."),
}

# Each fake "dimension" is a topic; a text points along the topics it mentions.
TOPICS = [("cook", "recipe", "dinner", "meal"), ("storage", "drive", "disk", "server"),
          ("film", "camera", "photo", "shot")]


class FakeModel:
    """Stands in for a SentenceTransformer: topic vectors, and a record of calls."""

    def __init__(self):
        self.calls = []

    def _embed(self, texts, normalize_embeddings=False):
        rows = [[sum(w in t.lower() for w in words) for words in TOPICS] for t in texts]
        out = np.array(rows, dtype=float) + 0.01  # never all zeros
        if normalize_embeddings:
            out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out

    def encode_document(self, texts, normalize_embeddings=False, **kwargs):
        self.calls.append("encode_document")
        return self._embed(texts, normalize_embeddings)

    def encode_query(self, texts, normalize_embeddings=False, **kwargs):
        self.calls.append("encode_query")
        out = self._embed(texts, normalize_embeddings)
        return out * 5 if not normalize_embeddings else out  # unnormalized: longer than 1

    def encode(self, *args, **kwargs):
        raise AssertionError("use encode_query/encode_document so the model's prompts are added")


@pytest.fixture
def retriever(tmp_path):
    return EmbeddingRetriever("fake", POSTS, model=FakeModel(), cache_dir=tmp_path)


def test_embedding_ranks_every_post_for_every_question(retriever):
    rankings = retriever.rank(["what's for dinner", "how much disk space"])
    assert len(rankings) == 2
    for ranking in rankings:
        assert sorted(ranking) == sorted(POSTS)


def test_embedding_best_match_ranks_first(retriever):
    # No question shares a word with its post; only the topic vectors match.
    rankings = retriever.rank(["a good meal", "a new disk", "which camera"])
    assert [r[0] for r in rankings] == ["recipes", "nas", "photos"]


def test_embedding_uses_the_query_prompt(retriever):
    retriever.rank(["a good meal"])
    assert retriever.model.calls[-1] == "encode_query"


def test_document_embeddings_are_cached(tmp_path):
    EmbeddingRetriever("fake", POSTS, model=FakeModel(), cache_dir=tmp_path)
    model = FakeModel()
    EmbeddingRetriever("fake", POSTS, model=model, cache_dir=tmp_path)
    assert "encode_document" not in model.calls
    changed = {**POSTS, "nas": mdx("My NAS now has twelve bays.")}
    EmbeddingRetriever("fake", changed, model=model, cache_dir=tmp_path)
    assert "encode_document" in model.calls  # new text, new cache entry


class Fixed:
    """A retriever that returns the same rankings whatever the questions."""

    def __init__(self, name, ranking):
        self.name, self.ranking = name, ranking

    def rank(self, questions):
        return [list(self.ranking) for _ in questions]


def test_rrf_by_hand():
    # k = 60:  a = 1/61 + 1/62 = 0.03252
    #          c = 1/63 + 1/61 = 0.03227
    #          b = 1/62 + 1/63 = 0.03200
    rrf = RRFRetriever([Fixed("x", ["a", "b", "c"]), Fixed("y", ["c", "a", "b"])])
    assert rrf.rank(["q1", "q2"]) == [["a", "c", "b"], ["a", "c", "b"]]


def test_rrf_post_missing_from_one_ranking():
    # b: 1/62 + 1/61 beats a: 1/61 + nothing from y
    rrf = RRFRetriever([Fixed("x", ["a", "b"]), Fixed("y", ["b"])])
    assert rrf.rank(["q"]) == [["b", "a"]]


def test_rrf_k_trades_one_top_rank_against_agreement():
    # a is 1st for x but 10th for y; b is 4th and 2nd.
    #   k = 0:   a = 1/1 + 1/10 = 1.10    b = 1/4 + 1/2 = 0.75      -> a first
    #   k = 60:  a = 1/61 + 1/70 = 0.0307 b = 1/64 + 1/62 = 0.0318  -> b first
    others = [f"p{i}" for i in range(8)]
    x = Fixed("x", ["a", *others[:2], "b", *others[2:]])
    y = Fixed("y", [others[0], "b", *others[1:], "a"])
    for k, first, second in [(0, "a", "b"), (60, "b", "a")]:
        ranking = RRFRetriever([x, y], k=k).rank(["q"])[0]
        assert ranking.index(first) < ranking.index(second), k


def test_rrf_name():
    assert RRFRetriever([Fixed("bm25", []), Fixed("gemma2", [])]).name == "rrf(bm25+gemma2)"
