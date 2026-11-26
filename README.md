# retrieval-bench

The code and data behind [Testing BM25 Against Embeddings on 49 Questions About My Blog](https://bryananthonio.com/blog/testing-bm25-against-embeddings-on-49-questions-about-my-blog/).

It compares seven retrievers on one question: given a question about my blog,
which of its 44 posts holds the answer?

- **BM25** ([bm25s](https://github.com/xhluca/bm25s))
- **Three embedding models**: [bge-m3](https://huggingface.co/BAAI/bge-m3),
  [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) and
  [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- **BM25 fused with each embedding model** by reciprocal rank fusion (RRF, k = 60)

## Results

Hit@1 and Hit@3 are the share of questions with a source post ranked first or
in the top 3; MRR is mean reciprocal rank. Each cell is direct wording /
paraphrase.

| Retriever            | Hit@1       | Hit@3       | MRR         |
| -------------------- | ----------- | ----------- | ----------- |
| Random               | 0.03 / 0.03 | 0.08 / 0.08 | 0.11 / 0.11 |
| BM25                 | 0.78 / 0.69 | 0.90 / 0.88 | 0.85 / 0.79 |
| bge-m3               | 0.80 / 0.78 | 0.92 / 0.92 | 0.87 / 0.85 |
| EmbeddingGemma 2     | 0.86 / 0.84 | 0.96 / 0.96 | 0.92 / 0.91 |
| Qwen3-Embedding-0.6B | 0.80 / 0.84 | 0.94 / 0.90 | 0.86 / 0.88 |
| RRF(BM25 + bge-m3)   | 0.86 / 0.82 | 1.00 / 0.94 | 0.91 / 0.88 |
| RRF(BM25 + Gemma 2)  | 0.86 / 0.82 | 1.00 / 0.94 | 0.91 / 0.88 |
| RRF(BM25 + Qwen3)    | 0.82 / 0.80 | 0.98 / 0.94 | 0.89 / 0.87 |

Every ranking, per retriever and question, is in `results/`. The notebook breaks
the scores down by question type and reads the questions where retrievers
disagree.

## What's here

```
corpus/                 the 44 posts, as MDX, as they were when the results were made
dataset/eval-set.yaml   the 49 questions, with labels, evidence quotes and paraphrases
results/                every retriever's ranking for every question, per wording
notebooks/              the analysis behind the post (reads results/, runs in seconds)
src/retrieval_bench/    retrievers, evaluation, the eval-set checker, analysis helpers
tests/
```

## The eval set

49 questions, each answerable from a single post:

- **single_fact (35):** one fact stated in one post.
- **current_state (8):** the answer changed over time. An older post has the
  outdated answer and only a newer one has the current answer. Retrievers
  don't look at dates, so these are the hardest.
- **opinion (6):** what I think about something, stated in one post.

The questions were drafted with Claude and then rewritten by hand to sound like
a reader asking. Each one lists its `source_posts` (every post containing the
answer; ranking any of them counts as a hit) and a verbatim `evidence` quote
from each, which the checker confirms is in the post. The header of
`dataset/eval-set.yaml` documents every field.

**Two wordings.** I wrote each question with its post in front of me, so it
tends to share the post's vocabulary, which is close to the best case for BM25.
So every question also has a `paraphrase`: the same question in a reader's
everyday words, written before any embedding results were seen and frozen
after. Every retriever runs on both wordings, and real queries probably fall
somewhere between them.

## How the retrievers see a post

- **Whole posts, no chunking.** Every post fits in all three models' context
  (the longest is about 7,600 tokens).
- **Post text** (`retrieval.post_text`) is the prose and code, plus the title,
  description, image alt text and captions from the frontmatter and components.
  Some answers exist only in a caption.
- **BM25 tokenization:** Lucene's 33 English stopwords and the Snowball stemmer.
  Lucene's list keeps words like "his", which matters here: questions say
  "Bryan… his" while posts are in the first person.

## Running it

Needs [uv](https://docs.astral.sh/uv/) and Python 3.14.

```sh
uv sync
uv run pytest
uv run check-eval-set                  # validate the eval set against corpus/
uv run run-eval                        # every retriever, direct wording
uv run run-eval --paraphrase           # every retriever, paraphrases
uv run run-eval --save                 # also write results/original.json (or paraphrase.json)
```

The first `run-eval` downloads the three embedding models from Hugging Face
(a few GB) and embeds every post. Post embeddings are cached in `cache/`.
`--blog PATH` runs against another folder of MDX posts.

Rerunning reproduces every Hit@k and MRR in `results/` exactly. A few
EmbeddingGemma 2 rankings can swap two neighboring posts far down the list
between runs (floating-point noise when embedding the questions); no metric
changes.

To open the analysis notebook:

```sh
uv run --group notebook --with jupyterlab jupyter lab notebooks/01-retrievers.ipynb
```

## License

The code is under the [MIT License](LICENSE). The posts in `corpus/` are
© Bryan Anthonio and licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see
[corpus/LICENSE](corpus/LICENSE).
