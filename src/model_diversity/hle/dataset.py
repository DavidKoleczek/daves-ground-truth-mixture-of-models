"""Humanity's Last Exam dataset loading and grading.

Adapted from frontier-model-relations project. Downloads the HLE dataset from
HuggingFace and provides access to questions and answers. Includes LLM-judge
grading via interop-router.
"""

from dataclasses import dataclass
from pathlib import Path
import random
import re
import shutil
from typing import cast

from huggingface_hub import hf_hub_download
from interop_router.router import Router
from interop_router.types import ChatMessage, RouterResponse, SupportedModel
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseInputFileParam,
    ResponseInputImageParam,
    ResponseInputTextParam,
)
import pyarrow.parquet as pq

_ContentPart = ResponseInputTextParam | ResponseInputImageParam | ResponseInputFileParam

_REPO_ID = "cais/hle"
_FILENAME = "data/test-00000-of-00001.parquet"

_JUDGE_SYSTEM_PROMPT = (
    "Judge whether the following [response] to [question] is correct or not "
    "based on the precise and unambiguous [correct_answer] below."
)

_JUDGE_USER_PROMPT_TEMPLATE = """[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. \
Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect \
based on [correct_answer], focusing only on if there are meaningful \
differences between [correct_answer] and the extracted_final_answer. \
Do not comment on any background to the problem, do not attempt to solve \
the problem, do not argue for any answer different than \
[correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the \
[correct_answer] given above, or is within a small margin of error for \
numerical problems. Answer 'no' otherwise, i.e. if there if there is any \
inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect."""


def _get_data_dir() -> Path:
    data_dir = Path(__file__).parents[3] / "tmp"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@dataclass(frozen=True)
class HLESample:
    id: str
    question: str
    answer: str
    answer_type: str
    image: str | None  # Base64 data URI or None


@dataclass(frozen=True)
class HLEGradeResult:
    correct: bool
    judge_response: str


def _download_dataset() -> Path:
    """Download the HLE parquet file from HuggingFace if not already present."""
    output_path = _get_data_dir() / "hle_test.parquet"
    if output_path.exists():
        return output_path

    cached_path = hf_hub_download(repo_id=_REPO_ID, filename=_FILENAME, repo_type="dataset")
    shutil.copy(cached_path, output_path)
    return output_path


def _parse_response_text(response: RouterResponse) -> str:
    """Extract and concatenate all assistant text content from a router response."""
    text_parts: list[str] = []
    for chat_message in response.output:
        message = chat_message.message
        if message.get("type") != "message":
            continue
        for part in message.get("content", []):
            if isinstance(part, dict) and part.get("type") == "output_text":
                text_parts.append(part.get("text", ""))
    return "".join(text_parts)


def load_hle_dataset(n: int | None = None, seed: int | None = None) -> list[HLESample]:
    """Load Humanity's Last Exam samples.

    Args:
        n: If provided, randomly sample this many items.
        seed: Random seed for reproducible sampling.

    Returns:
        List of HLESample objects containing questions and answers.
    """
    file_path = _download_dataset()
    table = pq.read_table(file_path)

    samples: list[HLESample] = []
    for i in range(table.num_rows):
        row = {col: table.column(col)[i].as_py() for col in table.column_names}
        samples.append(
            HLESample(
                id=row["id"],
                question=row["question"],
                answer=row["answer"],
                answer_type=row["answer_type"],
                image=row.get("image"),
            )
        )

    if n is not None:
        rng = random.Random(seed)
        samples = rng.sample(samples, min(n, len(samples)))

    return samples


async def grade_hle_response(
    router: Router,
    sample: HLESample,
    model_response: str,
    judge_model: str = "claude-sonnet-4-6",
) -> HLEGradeResult:
    """Grade a model's response to an HLE question using an LLM judge.

    Args:
        router: An interop-router Router instance with at least one provider registered.
        sample: The HLE sample containing the question and correct answer.
        model_response: The model's text response to grade.
        judge_model: Model to use as the judge.

    Returns:
        HLEGradeResult with correctness and the judge's full response.
    """
    user_prompt = _JUDGE_USER_PROMPT_TEMPLATE.format(
        question=sample.question,
        response=model_response,
        correct_answer=sample.answer,
    )

    content: list[_ContentPart] = [ResponseInputTextParam(type="input_text", text=user_prompt)]
    if sample.image:
        content.append(ResponseInputImageParam(type="input_image", detail="auto", image_url=sample.image))

    messages = [
        ChatMessage(message=EasyInputMessageParam(role="system", content=_JUDGE_SYSTEM_PROMPT)),
        ChatMessage(message=EasyInputMessageParam(role="user", content=content)),
    ]

    response = await router.create(
        input=messages,
        model=cast(SupportedModel, judge_model),
        max_output_tokens=32_000,
    )
    judge_text = _parse_response_text(response)

    match = re.search(r"correct:\s*(yes|no)", judge_text, re.IGNORECASE)
    correct = match.group(1).lower() == "yes" if match else False

    return HLEGradeResult(correct=correct, judge_response=judge_text)
