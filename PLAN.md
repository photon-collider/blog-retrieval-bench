# Plan

A blog series on retrieval-augmented Q&A over my own blog (44 MDX posts in
`~/Projects/personal-blog-astro/src/content/blog`).

- **Post 1:** compare retrievers.
- **Post 2:** add reranking.
- **Later, maybe:** abstaining on unanswerable questions; date-aware retrieval;
  embedding the photos themselves with a multimodal model (EmbeddingGemma 2).

## Decisions for post 1

- **Eval set:** `dataset/eval-set-v2-shortlist.yaml`, 54 hand-written questions
  (35 single_fact, 8 current_state, 5 multi_post, 6 opinion). Unanswerable
  questions are left out. Questions are worded the way a reader would ask.
- **Two wordings:** I wrote the questions with the post in mind, so they share
  its vocabulary; that's close to the best case for BM25. Each question also
  has a `paraphrase` with the same answer and labels but different vocabulary.
  Every retriever is run on both (`run-eval` and `run-eval --paraphrase`), and
  the post reports both as the two ends of a range. Real queries likely fall
  between them. Paraphrases are written as one reader: someone asking the
  blog's assistant who knows the names a post title would give them (Jev,
  Mealie, SvelteKit) but hasn't read the post, so they use everyday words, not
  the post's terms. Names stay; what's asked doesn't change. They are written
  before any embedding results are seen, without looking at per-question
  scores, and are frozen after that. The
  checker requires one per answerable question and flags any that are nearly
  the same as the question.
- **Tokenization:** `bm25s` with Lucene's 33 English stopwords and the
  Snowball stemmer, kept as the baseline. Lucene's list keeps "his", "he",
  "what" and "how". Questions say "Bryan… his" while posts are in the first
  person, so "his" has a high IDF and pulls toward posts about other people
  (sf75 ranks a Stephen King post first). That's an analysis note, not a fix;
  bm25s's 179-word `en_plus` list would remove it.
- **Retrievers:** BM25 (`bm25s`); three embedding models, bge-m3 (568M, a
  classic encoder), EmbeddingGemma 2 (multimodal, 740M total but a 270M text
  model, so say so when comparing) and Qwen3-Embedding-0.6B (from an LLM);
  and BM25 + each embedding model via reciprocal rank fusion (RRF). Whole
  posts, no chunking: every post fits all three models' context (longest
  ~7,600 tokens). Text only, though EmbeddingGemma 2 could embed images.
- **Post text:** what every retriever indexes is `retrieval.post_text`: the
  prose and code, plus the frontmatter title, description, imgAlt and
  imgCaption, and the alt/caption/title props of components. mp08's answer
  exists only in an image caption.
- **Labels:** one field, `source_posts`: every post that contains the answer
  (for multi_post, every post with part of it). There are no acceptable posts.
- **Metrics:** post-level Hit@k (any source post in the top k) and MRR for
  every question type; Recall@k for multi_post only, the one type that needs
  every source post. Reported per question type, for k = 1, 3 and 5.
  Random ranking scores Hit@1 0.03, Hit@5 0.15, MRR 0.12 on the shortlist
  (expected values over 44 posts, using each question's number of source posts).
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
4. [x] BM25 baseline (`retrievers.py`, `uv run run-eval`). Near the ceiling
       on single_fact and opinion; weak on current_state, where superseded
       posts often rank first. A rough run on `v1_paraphrase` dropped Hit@1
       from 0.78 to 0.51.
4b. [x] Paraphrases written blind by a subagent (persona and questions
       only), reviewed for meaning, frozen. BM25 on them: Hit@1 0.72, MRR
       0.81 (vs 0.78 and 0.85 original). The v1 paraphrases, which hid names,
       had dropped Hit@1 to 0.51, so most of that drop came from the
       obfuscation, not from realistic rewording.
5. [x] Embedding models (embeddings cached in `cache/`), then RRF, on both
       wordings. Original wording: RRF(BM25 + EmbeddingGemma 2) is best
       (Hit@1 0.87, Hit@3 1.00, MRR 0.92). Paraphrases: EmbeddingGemma 2 alone
       is best (Hit@1 0.83, MRR 0.90) and every hybrid falls slightly below
       it. current_state Hit@1 stays at 0.12–0.38 for every retriever.
6. [ ] Analysis notebook: 2×2 tables, disagreement questions, per-term
       breakdowns.
