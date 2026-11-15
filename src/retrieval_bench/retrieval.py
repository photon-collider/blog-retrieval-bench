"""Shared retrieval code for the blog Q&A benchmark.

Everything that decides what BM25 "sees" lives here, so the BM25 index, the
notebook, and any later analysis prepare and tokenize posts the same way.
"""
import re
from pathlib import Path

import bm25s
import Stemmer
import yaml

DEFAULT_BLOG = Path.home() / "Projects/personal-blog-astro/src/content/blog"

# Lucene's classic English stopword list (33 words), the default in Lucene and
# Elasticsearch. Note that it removes "no" and "not".
STOPWORDS = "en"
STEMMER = Stemmer.Stemmer("english")  # Snowball (Porter2)


def load_posts(blog: str | Path = DEFAULT_BLOG) -> dict[str, str]:
    """Post slug (filename without .mdx) -> the post's raw MDX text."""
    return {p.stem: p.read_text(encoding="utf-8") for p in Path(blog).glob("*.mdx")}


def strip_links(text: str) -> str:
    """Replace each Markdown link [text](url) with its text, so URLs don't become terms."""
    return re.sub(r"\[([^\]]*)\]\(\s*<?[^)\s>]*>?\s*\)", r"\1", text)


# Frontmatter keys and component props that hold text a reader sees. Everything
# else (image buckets, filenames, dates, sizes, chart data) is dropped.
FRONTMATTER_TEXT = ("title", "description", "imgAlt", "imgCaption")
PROP_TEXT = {"title", "alt", "caption", "imgAlt", "imgCaption"}

CODE_FENCE = re.compile(r"(^```.*?^```[^\n]*$)", re.DOTALL | re.MULTILINE)
COMPONENT_TAG = re.compile(r"<([A-Z][A-Za-z]*)\b(.*?)/?>", re.DOTALL)  # opening or self-closing
PROP = re.compile(r"""(\w+)=(?:"([^"]*)"|'([^']*)'|\{\s*"([^"]*)"\s*\}|\{\s*`([^`]*)`\s*\})""")
REMOVE = [
    re.compile(r"\{/\*.*?\*/\}", re.DOTALL),                                  # {/* MDX comment */}
    re.compile(r"""^import\b.*?(?:from\s+)?["'][^"']+["'];?[ \t]*$""", re.DOTALL | re.MULTILINE),
    re.compile(r"^export\s+function\b.*?^\}[ \t]*$", re.DOTALL | re.MULTILINE),       # multi-line function
    re.compile(r"^export\s+(?:const|let|var)\b.*$", re.MULTILINE),
    re.compile(r"</?[A-Z][A-Za-z]*>"),                                   # <Callout>, </Callout>
    re.compile(r"</?[a-z][a-z0-9]*(?:\s[^<>]*)?/?>"),                    # HTML tags: <td>, <code ...>
]


def _component_text(match: re.Match) -> str:
    """A component tag, reduced to the values of its text props."""
    texts = [next(v for v in values if v) for name, *values in PROP.findall(match.group(2))
             if name in PROP_TEXT and any(values)]
    return "\n".join(texts)


def _prose_text(prose: str) -> str:
    prose = COMPONENT_TAG.sub(_component_text, prose)
    for pattern in REMOVE:
        prose = pattern.sub("", prose)
    return strip_links(prose)


def post_text(raw: str) -> str:
    """The text every retriever indexes for one post.

    Keeps the prose, code blocks, tables and math as written, plus the
    frontmatter title, description, imgAlt and imgCaption, and the alt, caption
    and title props of components such as <BlogPostScreenshot>. Drops imports,
    exports, MDX comments, HTML tags, the rest of the frontmatter and component
    props, and link URLs (keeping the link text).
    """
    _, frontmatter, body = raw.split("---", 2)
    meta = yaml.safe_load(frontmatter)
    header = [str(meta[k]) for k in FRONTMATTER_TEXT if meta.get(k)]
    parts = CODE_FENCE.split(body)  # odd indexes are code fences, kept as written
    body = "".join(part if i % 2 else _prose_text(part) for i, part in enumerate(parts))
    text = "\n\n".join(header) + "\n\n" + body
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def tokenize(texts, return_ids=False):
    """Lowercase, split on words of 2+ characters, drop stopwords, then stem.

    Returns a list of token lists by default. With return_ids=True, returns the
    bm25s Tokenized object that bm25s.BM25.index() expects.
    """
    if isinstance(texts, str):
        texts = [texts]
    return bm25s.tokenize(texts, lower=True, stopwords=STOPWORDS, stemmer=STEMMER,
                          return_ids=return_ids, show_progress=False)
