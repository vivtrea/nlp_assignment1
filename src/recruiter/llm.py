"""OpenRouter (OpenAI-compatible) client with disk cache, pydantic validation and call log.

CLAUDE.md §6: prompts in prompts/*.md, one retry on validation error, temperature 0 for
scoring/eval, disk cache keyed by (model, prompt, input), every call logged.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from recruiter import config

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    pass


RETRY_DELAYS = (2, 5, 10, 20, 30)  # seconds; free-tier models return 429 under shared load

# Calls where an LLM component fell back to its rule-based version; evals report the rate.
STATS: dict[str, int] = {"llm_ok": 0, "fallbacks": 0}
LAST_ERROR: dict[str, str] = {}  # most recent LLMUnavailable reason, shown in the UI


def note(ok: bool) -> None:
    STATS["llm_ok" if ok else "fallbacks"] += 1


def reset_stats() -> None:
    STATS.update(llm_ok=0, fallbacks=0)


def load_prompt(name: str, **vars: Any) -> str:
    text = (config.PROMPTS_DIR / f"{name}.md").read_text()
    for k, v in vars.items():
        text = text.replace("{{" + k + "}}", str(v))
    return text


def _cache_path(model: str, prompt: str, temperature: float) -> Path:
    h = hashlib.sha256(f"{model}\x00{temperature}\x00{prompt}".encode()).hexdigest()
    return config.LLM_CACHE_DIR / h[:2] / f"{h}.json"


def _log(record: dict[str, Any]) -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (config.LOG_DIR / "llm_calls.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")


def complete(prompt: str, *, component: str, temperature: float = 0.0, model: str | None = None) -> str:
    """Raw completion with cache + logging. Raises LLMUnavailable without an API key."""
    if not config.llm_enabled():
        raise LLMUnavailable("OPENROUTER_API_KEY not set")
    model = model or config.LLM_MODEL
    cp = _cache_path(model, prompt, temperature)
    if cp.exists():
        out = json.loads(cp.read_text())["text"]
        _log({"component": component, "model": model, "cached": True, "ts": time.time()})
        return out

    import openai

    client = openai.OpenAI(base_url=config.LLM_BASE_URL, api_key=os.environ["OPENROUTER_API_KEY"], max_retries=0)
    t0 = time.time()
    resp = None
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            break
        except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError) as err:
            if attempt == len(RETRY_DELAYS):
                _log({"component": component, "model": model, "error": type(err).__name__, "ts": time.time()})
                LAST_ERROR["msg"] = f"{type(err).__name__} after {attempt + 1} attempts (model {model})"
                raise LLMUnavailable(LAST_ERROR["msg"]) from err
            time.sleep(RETRY_DELAYS[attempt])
        except openai.APIStatusError as err:  # 400/401/402/404...: retrying will not help
            _log({"component": component, "model": model, "error": f"{type(err).__name__} {err.status_code}", "ts": time.time()})
            LAST_ERROR["msg"] = f"{err.status_code} from OpenRouter for model {model!r}: {str(err.message)[:200]}"
            raise LLMUnavailable(LAST_ERROR["msg"]) from err
    text = (resp.choices[0].message.content or "") if resp and resp.choices else ""
    if not text.strip():
        raise LLMUnavailable("empty completion")
    usage = resp.usage
    _log(
        {
            "component": component,
            "model": model,
            "cached": False,
            "latency_s": round(time.time() - t0, 3),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "ts": time.time(),
        }
    )
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({"model": model, "text": text}))
    return text


def _extract_json(text: str) -> Any:
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1)
    start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
    if start < 0:
        raise ValueError("no JSON found")
    return json.loads(text[start : max(text.rfind("}"), text.rfind("]")) + 1])


def complete_json(
    prompt: str,
    schema: type[T],
    *,
    component: str,
    temperature: float = 0.0,
    model: str | None = None,
) -> T:
    """Completion parsed into `schema`; one retry with the validation error appended."""
    text = complete(prompt, component=component, temperature=temperature, model=model)
    try:
        return schema.model_validate(_extract_json(text))
    except (ValidationError, ValueError, json.JSONDecodeError) as err:
        retry = f"{prompt}\n\nYour previous answer was invalid: {str(err)[:500]}\nReturn ONLY valid JSON matching the schema."
        text = complete(retry, component=component, temperature=temperature, model=model)
        return schema.model_validate(_extract_json(text))
