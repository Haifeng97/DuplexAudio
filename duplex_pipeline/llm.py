from __future__ import annotations

import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .io import append_jsonl, atomic_write_json, iter_jsonl


def response_text(row: Dict[str, Any]) -> str:
    for key in ("response_text", "response", "output", "result", "content", "assistant_text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for nested_key in ("text", "content", "response_text"):
                nested = value.get(nested_key)
                if isinstance(nested, str) and nested.strip():
                    return nested.strip()
    return ""


def parse_json_response(text: str) -> Any:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    first_object = value.find("{")
    first_array = value.find("[")
    starts = [index for index in (first_object, first_array) if index >= 0]
    if starts:
        value = value[min(starts):]
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        parsed, _ = decoder.raw_decode(value)
        return parsed


class Dsv4Client:
    def __init__(self, config: Dict[str, Any]) -> None:
        api_key = os.environ.get(str(config.get("api_key_env") or "DSV4_API_KEY"), "")
        wsid = os.environ.get(str(config.get("wsid_env") or "DSV4_WSID"), "")
        if not api_key:
            raise RuntimeError(f"missing DSV4 API key env: {config.get('api_key_env', 'DSV4_API_KEY')}")
        if not wsid:
            raise RuntimeError(f"missing DSV4 Wsid env: {config.get('wsid_env', 'DSV4_WSID')}")
        try:
            import httpx  # type: ignore
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("DSV4 backend requires openai and httpx") from exc
        self.config = config
        self.client = OpenAI(
            api_key=api_key,
            base_url=str(config["base_url"]),
            default_headers={"Wsid": wsid},
            http_client=httpx.Client(timeout=float(config.get("timeout_sec", 300)), trust_env=False),
        )

    def close(self) -> None:
        self.client.close()

    def complete(self, request: Dict[str, Any]) -> str:
        query_id = f"duplex_{request['request_id']}_{uuid.uuid4()}"
        stream = self.client.chat.completions.create(
            model=str(self.config.get("model") or "cgame_aimate_generater"),
            messages=request["messages"],
            temperature=float(self.config.get("temperature", 0)),
            max_tokens=int(self.config.get("max_tokens", 4096)),
            stream=True,
            extra_body={
                "top_p": float(self.config.get("top_p", 1)),
                "top_k": int(self.config.get("top_k", 20)),
                "repetition_penalty": float(self.config.get("repetition_penalty", 1.0)),
                "thinking": bool(self.config.get("thinking", False)),
                "openai_infer": bool(self.config.get("openai_infer", True)),
                "query_id": query_id,
            },
        )
        chunks: List[str] = []
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        return "".join(chunks).strip()


def run_requests(
    config: Dict[str, Any],
    input_path: Path,
    output_path: Path,
    *,
    resume: bool,
    retries: int,
    progress_every: int,
    concurrency: int,
    quiet: bool = False,
) -> Dict[str, Any]:
    # Invoking llm-run is the explicit opt-in to the configured DSV4 endpoint.
    # Offline mode simply skips this command and fills the exported JSONL elsewhere.
    completed = set()
    if resume and output_path.exists():
        completed = {
            str(row.get("request_id") or "")
            for row in iter_jsonl(output_path)
            if row.get("status") == "ok"
        }
    elif output_path.exists():
        output_path.unlink()
    requests = list(iter_jsonl(input_path))
    pending = [row for row in requests if str(row.get("request_id") or "") not in completed]
    if concurrency <= 0:
        raise ValueError("concurrency must be > 0")
    client = Dsv4Client(dict(config["dsv4"]))
    ok = errors = 0
    started = time.monotonic()

    def execute(request: Dict[str, Any]) -> Dict[str, Any]:
        error = ""
        for attempt in range(retries + 1):
            try:
                text = client.complete(request)
                if not text:
                    raise RuntimeError("empty response")
                return {
                    "request_id": request["request_id"],
                    "job_type": request.get("job_type"),
                    "sample_id": request.get("sample_id"),
                    "status": "ok",
                    "response_text": text,
                    "attempt": attempt + 1,
                }
            except Exception as exc:
                error = repr(exc)
                if attempt < retries:
                    time.sleep(min(30.0, 2.0 ** attempt))
        return {
            "request_id": request["request_id"],
            "job_type": request.get("job_type"),
            "sample_id": request.get("sample_id"),
            "status": "error",
            "error": error,
        }

    if not quiet:
        print(
            f"LLM TOTAL: {len(completed)}/{len(requests)} "
            f"({100.0 * len(completed) / len(requests) if requests else 100.0:.2f}%) "
            f"pending={len(pending)} concurrency={concurrency}",
            flush=True,
        )
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(execute, request) for request in pending]
            for index, future in enumerate(as_completed(futures), start=1):
                result_row = future.result()
                append_jsonl(output_path, result_row)
                if result_row["status"] == "ok":
                    ok += 1
                else:
                    errors += 1
                elapsed = max(0.001, time.monotonic() - started)
                total_done = len(completed) + index
                rate = index / elapsed
                eta = (len(pending) - index) / rate if rate > 0 else 0.0
                if not quiet:
                    print(
                        f"\rLLM TOTAL: {total_done}/{len(requests)} "
                        f"({100.0 * total_done / len(requests) if requests else 100.0:.2f}%) "
                        f"ok={len(completed) + ok} errors={errors} rate={rate:.2f}/s eta={eta / 60:.1f}m",
                        end="",
                        flush=True,
                    )
                if progress_every > 0 and index % progress_every == 0:
                    pass
    finally:
        if pending and not quiet:
            print(flush=True)
        client.close()
    result = {
        "input": str(input_path), "output": str(output_path), "total": len(requests),
        "resumed": len(completed), "pending": len(pending), "ok": ok, "errors": errors,
        "concurrency": concurrency,
    }
    atomic_write_json(output_path.with_suffix(output_path.suffix + ".stats.json"), result)
    return result
