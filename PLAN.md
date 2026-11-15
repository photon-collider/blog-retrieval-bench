# Plan

A blog series on retrieval-augmented Q&A over my own blog (44 MDX posts in
`~/Projects/personal-blog-astro/src/content/blog`).

- **Post 1:** compare retrievers.
- **Post 2:** add reranking.
- **Later, maybe:** abstaining on unanswerable questions; date-aware retrieval.

## Decisions for post 1

- **Eval set:** `dataset/eval-set-v2-shortlist.yaml`, 54 hand-written questions
  (35 single_fact, 8 current_state, 5 multi_post, 6 opinion). Unanswerable
  questions are left out. Questions are worded the way a reader would ask;
  wording is not controlled or scored.
- **Retrievers:** BM25 (`bm25s`), EmbeddingGemma 2, Qwen3-Embedding-0.6B, and
  BM25 + each embedding model via reciprocal rank fusion (RRF). Whole posts,
  no chunking.
- **Post text:** what every retriever indexes is `retrieval.post_text`: the
  prose and code, plus the frontmatter title, description, imgAlt and
  imgCaption, and the alt/caption/title props of components. mp08's answer
  exists only in an image caption.
- **Metrics:** post-level Hit@k (a source or acceptable post counts), Recall@k
  (source posts only) and MRR, reported per question type, for k = 1, 3 and 5.
  Random ranking scores Hit@1 0.02, Hit@5 0.11, MRR 0.10 on the shortlist.
- **Analysis:** for each embedding model, a 2×2 table of BM25 hit/miss against
  embedding hit/miss, then read the questions where they disagree. A per-term
  BM25 breakdown (IDF × term-frequency part, per post) can explain specific
  misses such as cs04.
- **Not in post 1:** metrics about stale posts ranking above current ones; they
  belong to the reranking post.
- **The checker** (`uv run check-eval-set`) only validates the dataset. It
  reports nothing about retrieval, so it can't shape how questions are worded.

## Steps

1. [x] Package layout, Pydantic `Question` model, checker cleanup.
2. [x] `post_text`: turn an MDX post into the text retrievers index. The
       checker looks up evidence quotes in this text.
3. [x] Evaluation harness (`evaluate.py`, result models in `models.py`): run
       any retriever over the questions and compute the metrics per question
       type. Tested in `tests/test_evaluate.py`.
4. [ ] BM25 baseline.
5. [ ] Embedding models (embeddings cached in `cache/`), then RRF.
6. [ ] Analysis notebook: 2×2 tables, disagreement questions, per-term
       breakdowns.
