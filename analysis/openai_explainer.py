"""Optional structured OpenAI explanation for deterministic analysis results."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from .types import AnalysisError, AnalysisReport
from project_config import ENV_PATH


def _load_api_key(env_path: Path = ENV_PATH) -> str:
    """Load OPENAI_API_KEY from the environment or a local .env file.

    This intentionally uses only standard-library functionality. The .env
    file is never sent to OpenAI and the key is never included in prompts.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key

    if env_path.is_file():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "OPENAI_API_KEY":
                api_key = value.strip().strip("'\"")
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key
                    return api_key

    raise AnalysisError(f"Set OPENAI_API_KEY in {env_path}")


def _short_term_candidates(database_path: Path) -> list[dict[str, Any]]:
    """Build preservation-focused candidates using only SQLite data."""
    try:
        connection = sqlite3.connect(database_path)
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT *
            FROM model_portfolios
            WHERE Date = (SELECT MAX(Date) FROM model_portfolios)
            """
        ).fetchall()
    except sqlite3.Error as error:
        raise AnalysisError(f"Could not read portfolio database: {database_path}") from error
    finally:
        if "connection" in locals():
            connection.close()

    if not rows:
        raise AnalysisError("The SQLite database contains no portfolio rows")

    def number(row: sqlite3.Row, name: str) -> float | None:
        value = row[name]
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(str(row["Portfolio Name"]), []).append(row)

    candidates: list[dict[str, Any]] = []
    for name, assets in grouped.items():
        allocations = [number(row, "Allocation (%)") for row in assets]
        valid_allocations = [value for value in allocations if value is not None and value > 0]
        total = sum(valid_allocations)
        if total <= 0:
            continue

        def weighted_average(column: str) -> float | None:
            values = [number(row, column) for row in assets]
            pairs = [
                (allocation, value)
                for allocation, value in zip(allocations, values)
                if allocation is not None and allocation > 0 and value is not None
            ]
            return sum(allocation * value for allocation, value in pairs) / total if pairs else None

        weights = [allocation / total for allocation in valid_allocations]
        candidates.append({
            "portfolio_name": name,
            "asset_count": len(assets),
            "coverage": sum(
                1 for row in assets if number(row, "1 Year") is not None
            ) / len(assets),
            "ytd": weighted_average("YTD"),
            "one_year_return": weighted_average("1 Year"),
            "one_year_volatility": weighted_average("1Y Volatility"),
            "downside_risk": weighted_average("Downside Risk"),
            "maximum_drawdown": weighted_average("Maximum Drawdown"),
            "concentration": sum(weight * weight for weight in weights),
        })
    return candidates


def select_best_portfolio_from_sqlite(
    database_path: Path,
    model: str = "gpt-5.6-terra",
) -> dict[str, Any]:
    """Select one portfolio for short-term capital preservation.

    OpenAI receives only metrics calculated from ``database_path``. No market
    data, web search, or other external source is used.
    """
    api_key = _load_api_key()
    candidates = _short_term_candidates(database_path.expanduser().resolve())
    if not candidates:
        raise AnalysisError("No eligible portfolios were found")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Select exactly one portfolio for short-term capital "
                        "preservation. Prefer lower volatility, downside risk, "
                        "maximum drawdown, and concentration; then prefer higher "
                        "coverage and positive YTD/one-year return. Use only the "
                        "provided metrics. Return strict JSON with keys "
                        "portfolio_name and rationale."
                    ),
                },
                {"role": "user", "content": json.dumps(candidates, allow_nan=False)},
            ],
        )
    except ImportError as error:
        raise AnalysisError("Install the openai package to use AI selection") from error
    except Exception as error:
        raise AnalysisError(f"OpenAI portfolio selection failed: {error}") from error

    try:
        decision = json.loads(response.output_text)
        selected_name = decision["portfolio_name"]
        rationale = decision["rationale"]
    except (AttributeError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise AnalysisError("OpenAI returned an invalid portfolio selection") from error

    selected = next(
        (candidate for candidate in candidates
         if candidate["portfolio_name"] == selected_name),
        None,
    )
    if selected is None:
        raise AnalysisError(f"OpenAI selected an unknown portfolio: {selected_name!r}")
    return {"portfolio": selected, "rationale": rationale}


def _openai_client() -> tuple[Any, Any, Any, Any]:
    """Load dependencies, credentials, and return API helper types."""
    try:
        import openai
        from openai import OpenAI
        from pydantic import BaseModel
    except ImportError as error:
        raise AnalysisError("Install openai and pydantic to use AI analysis") from error

    # The key is read from .env and never printed, returned, or included in a
    # prompt. The OpenAI client reads it from the environment by default.
    return openai, OpenAI, BaseModel, _load_api_key()


def select_best_portfolio_with_openai(
    portfolios: list[dict[str, Any]],
    model: str,
) -> str:
    """Select a portfolio using short-term capital-preservation priorities.

    The model receives only portfolio summaries. It must return the exact
    portfolio name from that input; the caller remains responsible for loading
    the matching assets and producing the final report.
    """
    openai, OpenAI, BaseModel, api_key = _openai_client()

    class Selection(BaseModel):
        portfolio_name: str
        rationale: str

    prompt = {
        "task": "Select exactly one portfolio for short-term capital preservation.",
        "priorities": [
            "Minimize expected volatility.",
            "Prefer lower concentration.",
            "Prefer higher coverage, because unsupported assets increase uncertainty.",
            "Use expected return and score only after preservation criteria.",
            "Do not invent data or select a name absent from the candidates.",
        ],
        "candidates": portfolios,
    }
    try:
        response = OpenAI(api_key=api_key).responses.parse(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a conservative portfolio selector. Analyze only "
                        "the supplied metrics. This is research, not financial advice."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt, ensure_ascii=False, allow_nan=False),
                },
            ],
            text_format=Selection,
        )
    except openai.APIError as error:
        raise AnalysisError(f"OpenAI portfolio selection failed: {error}") from error

    if response.output_parsed is None:
        raise AnalysisError("OpenAI returned no portfolio selection")
    selected = response.output_parsed.portfolio_name
    valid_names = {portfolio["portfolio_name"] for portfolio in portfolios}
    if selected not in valid_names:
        raise AnalysisError(
            f"OpenAI selected an unknown portfolio: {selected!r}"
        )
    return selected


def explain_with_openai(
    report: AnalysisReport,
    model: str,
) -> dict[str, Any]:
    """Explain—but never alter—the deterministic recommendation."""
    try:
        import openai
        from openai import OpenAI
        from openai.types.responses import EasyInputMessageParam
        from openai.types.shared_params import Reasoning
        from pydantic import BaseModel
    except ImportError as error:
        raise AnalysisError("Install the openai package to use --explain") from error

    api_key = _load_api_key()

    class Explanation(BaseModel):
        summary: str
        best_medium_term_portfolio: str
        key_reasons: list[str]
        allocation_commentary: list[str]
        risks_and_limitations: list[str]
        disclaimer: str

    payload = report.to_dict()
    payload["explanation"] = None
    reasoning: Reasoning = {"effort": "medium"}
    messages: list[EasyInputMessageParam] = [
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
    ]
    try:
        response = OpenAI(api_key=api_key).responses.parse(
            model=model,
            reasoning=reasoning,
            input=messages,
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
