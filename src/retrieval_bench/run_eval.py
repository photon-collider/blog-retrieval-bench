"""Run retrievers over the eval set and print their scores per question type.

Usage: uv run run-eval [--file dataset/eval-set-v2-shortlist.yaml] [--blog PATH]
"""
import argparse

from retrieval_bench.evaluate import evaluate, print_summaries, summarize
from retrieval_bench.models import EVAL_SET, load_questions
from retrieval_bench.retrieval import DEFAULT_BLOG, load_posts
from retrieval_bench.retrievers import BM25Retriever


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blog", default=str(DEFAULT_BLOG))
    ap.add_argument("--file", default=str(EVAL_SET))
    a = ap.parse_args()
    posts = load_posts(a.blog)
    questions = load_questions(a.file)

    retrievers = [BM25Retriever(posts)]  # embedding models and RRF join here in step 5
    results = [r for retriever in retrievers for r in evaluate(retriever, questions)]
    print_summaries(summarize(results))


if __name__ == "__main__":
    main()
