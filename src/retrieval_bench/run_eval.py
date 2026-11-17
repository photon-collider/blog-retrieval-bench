"""Run retrievers over the eval set and print their scores per question type.

Usage: uv run run-eval [--file dataset/eval-set-v2-shortlist.yaml] [--blog PATH] [--paraphrase] [--save]

--paraphrase asks each question in its reworded form (the `paraphrase` field).
--save also writes every result, rankings included, to results/{wording}.json
for the analysis notebook.
"""
import argparse
import sys

from retrieval_bench.evaluate import evaluate, print_summaries, summarize
from retrieval_bench.models import EVAL_SET, RESULTS_DIR, load_questions, paraphrased, save_results
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
    ap.add_argument("--save", action="store_true", help="write results/{wording}.json")
    a = ap.parse_args()
    posts = load_posts(a.blog)
    questions = load_questions(a.file)
    if a.paraphrase:
        try:
            questions = paraphrased(questions)
        except ValueError as e:
            sys.exit(str(e))
    wording = "paraphrase" if a.paraphrase else "original"
    print(f"wording: {wording}\n")

    bm25 = BM25Retriever(posts)
    embedders = [EmbeddingRetriever(name, posts) for name in EMBEDDING_MODELS]
    retrievers = [bm25, *embedders, *(RRFRetriever([bm25, e]) for e in embedders)]
    results = [r for retriever in retrievers for r in evaluate(retriever, questions)]
    print_summaries(summarize(results))
    if a.save:
        path = RESULTS_DIR / f"{wording}.json"
        save_results(path, results)
        print(f"\nsaved {len(results)} results to {path.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
