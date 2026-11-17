"""Retrievers. Each has a `name` and a `rank` method, so evaluate.evaluate() can
run it (see evaluate.Retriever).

Every retriever indexes retrieval.post_text(raw) for each post, so they all see
the same text.

- BM25Retriever: keyword matching.
- EmbeddingRetriever: an embedding model (see EMBEDDING_MODELS), cosine similarity.
- RRFRetriever: fuses the rankings of other retrievers.
"""

import hashlib
from pathlib import Path

import numpy as np
from bm25s import BM25

from retrieval_bench.evaluate import Retriever
from retrieval_bench.models import ROOT, Slug
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


# --- Embedding models -------------------------------------------------------

CACHE_DIR = ROOT / "cache"

# name -> Hugging Face model id. encode_query and encode_document add the
# "query" and "document" prompts from each model's sentence-transformers config:
#   bge-m3            none (it's trained without prompts)
#   EmbeddingGemma 2  query "task: search result | query: ", document "title: none | text: "
#   Qwen3-Embedding   query "Instruct: Given a web search query, retrieve relevant
#                     passages that answer the query\nQuery:", document "" (none)
# Context: 8,192 tokens for bge-m3 and EmbeddingGemma 2, 32,768 for Qwen3. The
# longest post is about 7,600 tokens in any of their tokenizers, so none is truncated.
# Sizes: bge-m3 568M; EmbeddingGemma 2 740M, of which the text model is 270M
# (the rest are image and audio encoders, unused here); Qwen3 600M.
EMBEDDING_MODELS = {
    "bge-m3": "BAAI/bge-m3",
    "gemma2": "google/embeddinggemma-2",
    "qwen3-0.6b": "Qwen/Qwen3-Embedding-0.6B",
}


def load_embedding_model(model_id: str):
    """A SentenceTransformer, imported here so BM25-only code doesn't load torch."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id)
    # EmbeddingGemma 2's config sets no limit, so a text past the 8,192 tokens it
    # was trained on would go in untruncated. Cap every model there; no post is
    # that long today, so this changes no embedding.
    model.max_seq_length = min(model.max_seq_length, 8192)
    return model


def document_embeddings(model, model_id: str, texts: list[str], cache_dir: Path = CACHE_DIR) -> np.ndarray:
    """One unit-length row per text, embedded with the model's document prompt.

    Cached in cache_dir under the model id and a hash of every text, so a change
    to post_text (or a new post) re-embeds automatically.
    """
    digest = hashlib.sha256("\0".join([model_id, *texts]).encode()).hexdigest()[:16]
    path = cache_dir / f"{model_id.replace('/', '__')}-{digest}.npy"
    if path.exists():
        return np.load(path)
    # One post per batch: a batch is padded to its longest text, and 32 posts of
    # up to ~7,600 tokens run out of GPU memory on a laptop.
    embeddings = model.encode_document(
        texts, batch_size=1, normalize_embeddings=True, convert_to_numpy=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    return embeddings


class EmbeddingRetriever:
    """Dense retrieval over whole posts: embed every post once (cached), embed
    each question, rank posts by cosine similarity.

        retriever = EmbeddingRetriever("gemma2", load_posts())
        retriever.rank(["What machine runs Bryan's recipe manager?"])

    `model` is for tests: anything with encode_query and encode_document.
    """

    def __init__(self, name: str, posts: dict[Slug, str], model=None, cache_dir: Path = CACHE_DIR) -> None:
        self.name = name
        model_id = EMBEDDING_MODELS.get(name, name)
        self.model = model if model is not None else load_embedding_model(model_id)
        self.slugs = list(posts)  # row i of doc_embeddings is self.slugs[i]
        self.doc_embeddings = document_embeddings(
            self.model, model_id, [post_text(posts[slug]) for slug in self.slugs], cache_dir)

    def rank(self, questions: list[str]) -> list[list[Slug]]:
        """For each question, every post's slug, most similar first."""

        emb_qs = self.model.encode_query(questions, convert_to_numpy=True)
        output: list[list[Slug]] = []

        for q in emb_qs:
            scores = []
            for i,doc_emb in enumerate(self.doc_embeddings):
                slug = self.slugs[i]
                score = q @ doc_emb
                scores.append((slug, score))

            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            ranking = [slug for slug,_ in scores]
            output.append(ranking)

        return output


# --- Reciprocal rank fusion ---------------------------------------------------


class RRFRetriever:
    """Fuses other retrievers' rankings with reciprocal rank fusion: each post
    scores the sum, over the retrievers, of 1 / (k + its rank there), with rank
    1 at the top. A post a retriever left out adds nothing from that retriever.
    k = 60 is the usual default (Cormack et al., 2009).

        RRFRetriever([bm25, gemma]).rank(questions)
    """

    def __init__(self, retrievers: list[Retriever], k: int = 60) -> None:
        self.retrievers = retrievers
        self.k = k
        self.name = "rrf(" + "+".join(r.name for r in retrievers) + ")"

    def rank(self, questions: list[str]) -> list[list[Slug]]:
        """For each question, every post any retriever ranked, best fused score first."""
        # TODO: get each retriever's rankings for all the questions.
        ret_rankings = []

        for r in self.retrievers:
            rankings = r.rank(questions)
            ret_rankings.append(rankings)
        
        output = []

        for i in range(len(questions)):
            slug_scores = {}
            for j in range(len(self.retrievers)):
                q_rankings = ret_rankings[j][i]
                for rank,slug in enumerate(q_rankings, start = 1):
                    slug_scores[slug] = slug_scores.get(slug, 0) + 1/(self.k + rank)

            scores = [ (slug, score) for (slug,score) in slug_scores.items()]
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            output_ranking = [slug for slug,_ in scores]
            output.append(output_ranking)

        return output
        
            
