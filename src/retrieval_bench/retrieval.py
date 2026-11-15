"""Shared retrieval code for the blog Q&A benchmark.

Everything that decides what BM25 "sees" lives here, so the BM25 index, the
notebook, and check_eval_set.py's word-overlap measure all tokenize the same way.
"""
import bm25s
import Stemmer

# Lucene's classic English stopword list (33 words), the default in Lucene and
# Elasticsearch. Note that it removes "no" and "not".
STOPWORDS = "en"
STEMMER = Stemmer.Stemmer("english")  # Snowball (Porter2)


def tokenize(texts, return_ids=False):
    """Lowercase, split on words of 2+ characters, drop stopwords, then stem.

    Returns a list of token lists by default. With return_ids=True, returns the
    bm25s Tokenized object that bm25s.BM25.index() expects.
    """
    if isinstance(texts, str):
        texts = [texts]
    return bm25s.tokenize(texts, lower=True, stopwords=STOPWORDS, stemmer=STEMMER,
                          return_ids=return_ids, show_progress=False)
