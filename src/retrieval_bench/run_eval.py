"""Run retrievers over the eval set and print their scores per question type.

Usage: uv run run-eval [--file dataset/eval-set-v2-shortlist.yaml] [--blog PATH] [--paraphrase]

--paraphrase asks each question in its reworded form (the `paraphrase` field).
"""
import argparse
import sys

from retrieval_bench.evaluate import evaluate, print_summaries, summarize
from retrieval_bench.models import EVAL_SET, load_questions, paraphrased
from retrieval_bench.retrieval import DEFAULT_BLOG, load_posts
from retrieval_bench.retrievers import (
    EMBEDDING_MODELS,
    BM25Retriever,
    EmbeddingRetriever,
    RRFRetriever,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blog", default=str(DEFAULT_BLOG))
    ap.add_argument("--file", default=str(EVAL_SET))
    ap.add_argument("--paraphrase", action="store_true",
                    help="use each question's paraphrase instead of its original wording")
    a = ap.parse_args()
    posts = load_posts(a.blog)
    questions = load_questions(a.file)
    if a.paraphrase:
        try:
            questions = paraphrased(questions)
        except ValueError as e:
            sys.exit(str(e))
    print(f"wording: {'paraphrase' if a.paraphrase else 'original'}\n")

    bm25 = BM25Retriever(posts)
    embedders = [EmbeddingRetriever(name, posts) for name in EMBEDDING_MODELS]
    retrievers = [bm25, *embedders, *(RRFRetriever([bm25, e]) for e in embedders)]
    results = [r for retriever in retrievers for r in evaluate(retriever, questions)]
    print_summaries(summarize(results))


if __name__ == "__main__":
    main()
