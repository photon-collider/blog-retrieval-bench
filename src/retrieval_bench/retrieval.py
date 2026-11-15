"""Shared retrieval code for the blog Q&A benchmark.

Everything that decides what BM25 "sees" lives here, so the BM25 index, the
notebook, and any later analysis prepare and tokenize posts the same way.
"""
import re
from pathlib import Path

import bm25s
import Stemmer

DEFAULT_BLOG = Path.home() / "Projects/personal-blog-astro/src/content/blog"

# Lucene's classic English stopword list (33 words), the default in Lucene and
# Elasticsearch. Note that it removes "no" and "not".
STOPWORDS = "en"
STEMMER = Stemmer.Stemmer("english")  # Snowball (Porter2)


def load_posts(blog: str | Path = DEFAULT_BLOG) -> dict[str, str]:
    """Post slug (filename without .mdx) -> the post's raw MDX text."""
    return {p.stem: p.read_text(encoding="utf-8") for p in Path(blog).glob("*.mdx")}


def strip_links(text: str) -> str:
    """Replace each Markdown link [text](url) with its text, so URLs don't become terms.
    Apply this before tokenize when indexing a post."""
    return re.sub(r"\[([^\]]*)\]\(\s*<?[^)\s>]*>?\s*\)", r"\1", text)


def tokenize(texts, return_ids=False):
    """Lowercase, split on words of 2+ characters, drop stopwords, then stem.

    Returns a list of token lists by default. With return_ids=True, returns the
    bm25s Tokenized object that bm25s.BM25.index() expects.
    """
    if isinstance(texts, str):
        texts = [texts]
    return bm25s.tokenize(texts, lower=True, stopwords=STOPWORDS, stemmer=STEMMER,
                          return_ids=return_ids, show_progress=False)
