"""Pydantic models for the project, and loading the eval set into them.

Used by check_eval_set.py, which validates the eval-set files, and by the
evaluation harness and notebooks, which run retrievers over the questions.
"""
import collections
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

Slug = str  # a post's filename without .mdx, e.g. "my-homelab"

QuestionType = Literal["single_fact", "multi_post", "current_state", "opinion", "unanswerable"]
Topic = Literal["ai-llm", "ml-nlp", "backend", "web-design", "homelab", "self-hosting",
                "computing", "personal", "writing-productivity", "cross-topic"]
Probe = Literal["vocab_mismatch", "lexical_collision", "sibling_confusion", "near_miss",
                "caption_only", "alt_text_only", "table_only", "code_only", "callout_only",
                "buried_detail", "short_post", "long_post", "recency", "false_premise",
                "aggregation", "multi_hop"]

POST_FIELDS = ("source_posts", "superseded_posts", "related_posts")
ROOT = Path(__file__).resolve().parents[2]  # project root (src/retrieval_bench/ is two levels down)
EVAL_SET = ROOT / "dataset" / "eval-set-v2-shortlist.yaml"
RESULTS_DIR = ROOT / "results"  # run-eval --save writes here; the notebook reads it


class Question(BaseModel):
    """One eval-set item, as written in the YAML. The field meanings are in the
    YAML file's header comment.

    Parsing checks each field on its own: required fields are present, `type`,
    `topic` and `probe` use an allowed value, and lists hold strings.
    check_eval_set.validate() checks the rules that span fields or need the posts.
    """
    model_config = ConfigDict(extra="forbid")  # a misspelled field name is an error

    id: str
    type: QuestionType
    topic: Topic
    probe: Probe | None = None
    question: str
    source_posts: list[Slug] = []
    superseded_posts: list[Slug] = []
    related_posts: list[Slug] = []
    evidence: dict[Slug, str] = {}
    reference_answer: str = Field(min_length=1)
    key_facts: list[str] = []
    notes: str | None = None
    paraphrase: str | None = None  # the same question reworded; see paraphrased()

    # Reference only; not used by any check
    v1_question: str | None = None
    v1_paraphrase: str | None = None
    in_v1_subset: bool = False


def load_items(path: str | Path) -> list[dict[str, Any]]:
    """Every item in the file as a plain dict, blank `question` or not."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["questions"]

def is_blank(item: dict[str, Any]) -> bool:
    """A blank `question` marks a workbook item that wasn't chosen."""
    return not (item.get("question") or "").strip()

def parse_questions(items: list[dict[str, Any]]) -> tuple[list[Question], dict[str, list[str]]]:
    """Parse each item into a Question. Returns the ones that parsed, and for the
    ones that didn't, id -> one "field: problem" message per problem."""
    questions, errors = [], collections.defaultdict(list)
    for item in items:
        try:
            questions.append(Question.model_validate(item))
        except ValidationError as e:
            qid = str(item.get("id", "?"))
            for err in e.errors():
                field = ".".join(str(part) for part in err["loc"])
                errors[qid].append(f"{field}: {err['msg']}")
    return questions, dict(errors)

def load_questions(path: str | Path = EVAL_SET) -> list[Question]:
    """The non-blank questions in `path`. Raises ValueError if any fail to parse;
    run `uv run check-eval-set` to see every problem."""
    questions, errors = parse_questions([i for i in load_items(path) if not is_blank(i)])
    if errors:
        raise ValueError(f"{len(errors)} questions in {path} failed to parse: {sorted(errors)}")
    return questions


def paraphrased(questions: list[Question]) -> list[Question]:
    """The same questions with `question` replaced by `paraphrase`, to run the
    eval on the reworded set. Unanswerable questions are left as they are,
    since evaluate() skips them. Raises ValueError if an answerable question
    has no paraphrase."""
    missing = [q.id for q in questions if q.source_posts and not q.paraphrase]
    if missing:
        raise ValueError(f"no paraphrase for {missing}; run `uv run check-eval-set`")
    return [q.model_copy(update={"question": q.paraphrase}) if q.paraphrase else q
            for q in questions]


class QuestionResult(BaseModel):
    """How one retriever did on one question. Made by evaluate.evaluate()."""
    model_config = ConfigDict(frozen=True)

    retriever: str
    question_id: str
    type: QuestionType
    ranking: list[Slug]               # the retriever's ranking, best first
    first_source_rank: int | None     # 1 = top; None if no source post was ranked
    hit_at: dict[int, bool]           # k -> is a source post in the top k?
    recall_at: dict[int, float] | None  # k -> share of source posts in the top k;
                                        # multi_post only, None for other types

    @property
    def reciprocal_rank(self) -> float:
        """1 / first_source_rank, or 0 if no source post was ranked."""
        return 0.0 if self.first_source_rank is None else 1 / self.first_source_rank


class Summary(BaseModel):
    """One retriever's averages over a group of questions. Made by evaluate.summarize()."""
    model_config = ConfigDict(frozen=True)

    retriever: str
    group: str                        # a question type, or "all"
    n: int                            # questions in the group
    hit_at: dict[int, float]          # k -> Hit@k: share of questions with a hit in the top k
    recall_at: dict[int, float] | None  # k -> Recall@k, averaged; multi_post only
    mrr: float                        # mean reciprocal rank


_RESULTS = TypeAdapter(list[QuestionResult])

def save_results(path: str | Path, results: list[QuestionResult]) -> None:
    """Write results as JSON, rankings included, so analysis needs no models."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(_RESULTS.dump_json(results, indent=1))

def load_results(path: str | Path) -> list[QuestionResult]:
    """Read results written by save_results."""
    return _RESULTS.validate_json(Path(path).read_bytes())
