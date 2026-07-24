"""Optional structured OpenAI explanation for deterministic analysis results."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

from .types import AnalysisError, AnalysisReport
from project_config import ENV_PATH


def explain_with_openai(
    report: AnalysisReport,
    model: str,
) -> dict[str, Any]:
    """Explain—but never alter—the deterministic recommendation."""
    try:
        import openai
        from openai import OpenAI
        from pydantic import BaseModel
    except ImportError as error:
        raise AnalysisError("Install the openai package to use --explain") from error

    load_dotenv(ENV_PATH, override=False)
    if not os.environ.get("OPENAI_API_KEY"):
        raise AnalysisError(
            f"Set OPENAI_API_KEY in {ENV_PATH} when --explain is used"
        )

    class Explanation(BaseModel):
        summary: str
        best_medium_term_portfolio: str
        key_reasons: list[str]
        allocation_commentary: list[str]
        risks_and_limitations: list[str]
        disclaimer: str

    payload = report.to_dict()
    payload["explanation"] = None
    try:
        response = OpenAI().responses.parse(
            model=model,
            reasoning={"effort": "medium"},
            input=[
                {
                    "role": "system",
                    "content": (
                        "Explain only the provided deterministic investment "
                        "analysis. Do not change rankings, scores, or allocations. "
                        "Do not promise future performance. State that this is "
                        "research, not personalized financial advice."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                },
            ],
            text_format=Explanation,
        )
    except openai.AuthenticationError as error:
        raise AnalysisError(
            "OpenAI rejected the API key. Check OPENAI_API_KEY in .env and "
            "confirm that the key belongs to the intended API project."
        ) from error
    except openai.RateLimitError as error:
        body = error.body if isinstance(error.body, dict) else {}
        if body.get("code") == "insufficient_quota":
            raise AnalysisError(
                "OpenAI API quota is unavailable. Add API credits or raise the "
                "organization/project spend limit, then retry. You can run "
                "without --explain to generate the local quantitative report."
            ) from error
        raise AnalysisError(
            "OpenAI rate limit reached. Wait briefly and retry, or run without "
            "--explain."
        ) from error
    except openai.APITimeoutError as error:
        raise AnalysisError(
            "The OpenAI request timed out. Retry or run without --explain."
        ) from error
    except openai.APIConnectionError as error:
        raise AnalysisError(
            "Could not connect to OpenAI. Check the network connection or run "
            "without --explain."
        ) from error
    except openai.APIStatusError as error:
        raise AnalysisError(
            f"OpenAI API request failed with HTTP {error.status_code}. "
            "Check model access and project permissions, or run without --explain."
        ) from error
    except openai.APIError as error:
        raise AnalysisError(
            "OpenAI API request failed. Retry or run without --explain."
        ) from error

    if response.output_parsed is None:
        raise AnalysisError("OpenAI returned no structured explanation")
    return response.output_parsed.model_dump()
