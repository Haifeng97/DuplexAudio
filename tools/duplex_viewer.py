#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import array
import json
import mimetypes
import os
import posixpath
import re
import sys
import wave
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "duplex_viewer_static"
WAVEFORM_CACHE: Dict[Tuple[str, int, int, int], Dict[str, Any]] = {}


def compact_text(value: Any, limit: int = 120) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def label_counts(timeline: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ent in timeline:
        label = str(ent.get("label") or ent.get("token_text") or "")
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:80])


def source_counts(timeline: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ent in timeline:
        source = str(ent.get("audio_source") or "")
        counts[source] = counts.get(source, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def timeline_groups(timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    for ent in timeline:
        key = (
            ent.get("label"),
            ent.get("kind"),
            ent.get("audio_source"),
            ent.get("turn_id"),
            ent.get("label_type"),
        )
        if groups and groups[-1]["_key"] == key:
            groups[-1]["end_idx"] = ent.get("idx")
            groups[-1]["end_sec"] = ent.get("end_sec")
            groups[-1]["end_sample"] = ent.get("end_sample")
            groups[-1]["chunks"] += 1
            if ent.get("label_type") == "text":
                groups[-1]["text"] += str(ent.get("token_text") or ent.get("label") or "")
            continue
        groups.append(
            {
                "_key": key,
                "start_idx": ent.get("idx"),
                "end_idx": ent.get("idx"),
                "chunks": 1,
                "label": ent.get("label"),
                "label_type": ent.get("label_type"),
                "kind": ent.get("kind"),
                "audio_source": ent.get("audio_source"),
                "turn_id": ent.get("turn_id"),
                "start_sec": ent.get("start_sec"),
                "end_sec": ent.get("end_sec"),
                "start_sample": ent.get("start_sample"),
                "end_sample": ent.get("end_sample"),
                "text": str(ent.get("token_text") or "") if ent.get("label_type") == "text" else "",
            }
        )
    for group in groups:
        group.pop("_key", None)
    return groups


@dataclass
class ManifestIndex:
    path: Path
    offsets: List[int]
    summaries: List[Dict[str, Any]]
    scenario_counts: Dict[str, int]
    total_duration_sec: float

    @classmethod
    def build(cls, path: Path) -> "ManifestIndex":
        offsets: List[int] = []
        summaries: List[Dict[str, Any]] = []
        scenario_counts: Dict[str, int] = {}
        total_duration_sec = 0.0
        with path.open("rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                offsets.append(offset)
                row = json.loads(line.decode("utf-8", errors="replace"))
                scenario = str(row.get("scenario") or "")
                stats = row.get("stats") or {}
                duration = float(stats.get("duration_sec") or 0.0)
                total_duration_sec += duration
                scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
                summaries.append(
                    {
                        "index": len(summaries),
                        "id": row.get("id"),
                        "scenario": scenario,
                        "duration_sec": duration,
                        "timeline_chunks": stats.get("timeline_chunks"),
                        "sample_rate": row.get("sample_rate"),
                        "chunk_ms": row.get("chunk_ms"),
                        "question_text": compact_text(row.get("question_text"), 160),
                        "answer_text": compact_text(row.get("answer_text"), 160),
                        "audio": row.get("audio"),
                    }
                )
        return cls(path, offsets, summaries, scenario_counts, total_duration_sec)

    def read_row(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= len(self.offsets):
            raise IndexError(index)
        with self.path.open("rb") as f:
            f.seek(self.offsets[index])
            return json.loads(f.readline().decode("utf-8", errors="replace"))

    def as_dict(self, key: int) -> Dict[str, Any]:
        count = len(self.offsets)
        return {
            "key": key,
            "path": safe_rel(self.path),
            "count": count,
            "scenario_counts": self.scenario_counts,
            "avg_duration_sec": round(self.total_duration_sec / count, 4) if count else 0,
        }


class AppState:
    def __init__(self, manifests: List[Path]):
        self.manifests = [ManifestIndex.build(p) for p in manifests]

    def manifest(self, key: int) -> ManifestIndex:
        if key < 0 or key >= len(self.manifests):
            raise IndexError(key)
        return self.manifests[key]


def discover_manifests(explicit: List[str], *, auto_discover: bool = True) -> List[Path]:
    found: List[Path] = []
    for value in explicit:
        path = resolve_project_path(value)
        if path.is_file() and path not in found:
            found.append(path)
    if not auto_discover:
        return found
    patterns = [
        "outputs/pipeline_real_*_20/manifest.jsonl",
        "outputs/pipeline_mock_*_200/manifest.jsonl",
        "outputs/**/manifest.jsonl",
    ]
    for pattern in patterns:
        for path in sorted(ROOT.glob(pattern)):
            resolved = path.resolve()
            if resolved.is_file() and resolved not in found:
                found.append(resolved)
    return found


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def parse_int(params: Dict[str, List[str]], name: str, default: int = 0) -> int:
    try:
        return int((params.get(name) or [str(default)])[0])
    except ValueError:
        return default


def row_audio_path(row: Dict[str, Any]) -> Path:
    audio_path = resolve_project_path(str(row.get("audio") or ""))
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    try:
        audio_path.relative_to(ROOT)
    except ValueError as exc:
        raise PermissionError(f"audio path outside project root: {audio_path}") from exc
    return audio_path


def waveform_payload(audio_path: Path, width: int = 1600) -> Dict[str, Any]:
    width = max(64, min(4096, int(width)))
    st = audio_path.stat()
    cache_key = (str(audio_path), st.st_mtime_ns, st.st_size, width)
    cached = WAVEFORM_CACHE.get(cache_key)
    if cached:
        return cached

    with wave.open(str(audio_path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frame_count = wf.getnframes()
        raw = wf.readframes(frame_count)

    if sample_width != 2:
        raise ValueError(f"unsupported sample width for waveform: {sample_width}")

    samples = array.array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()

    bucket_frames = max(1, (frame_count + width - 1) // width)
    peaks: List[List[float]] = []
    scale = 32768.0
    for bucket_start in range(0, frame_count, bucket_frames):
        bucket_end = min(frame_count, bucket_start + bucket_frames)
        start = bucket_start * channels
        end = bucket_end * channels
        lo = 0
        hi = 0
        for sample in samples[start:end]:
            if sample < lo:
                lo = sample
            if sample > hi:
                hi = sample
        peaks.append([round(lo / scale, 4), round(hi / scale, 4)])

    payload = {
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frame_count,
        "duration_sec": round(frame_count / sample_rate, 6) if sample_rate else 0,
        "width": width,
        "peaks": peaks,
    }
    WAVEFORM_CACHE[cache_key] = payload
    if len(WAVEFORM_CACHE) > 256:
        WAVEFORM_CACHE.pop(next(iter(WAVEFORM_CACHE)))
    return payload


def item_payload(manifest_key: int, manifest: ManifestIndex, index: int) -> Dict[str, Any]:
    row = manifest.read_row(index)
    timeline = row.get("timeline") or []
    row["_viewer"] = {
        "manifest_path": safe_rel(manifest.path),
        "index": index,
        "audio_url": f"/api/audio?manifest={manifest_key}&index={index}",
        "waveform_url": f"/api/waveform?manifest={manifest_key}&index={index}&width=1600",
        "label_counts": label_counts(timeline),
        "source_counts": source_counts(timeline),
        "timeline_groups": timeline_groups(timeline),
    }
    return row


class Handler(BaseHTTPRequestHandler):
    server_version = "DuplexViewer/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))

    def send_json(self, payload: Any, status: int = 200) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, message: str, status: int = 400) -> None:
        self.send_json({"error": message}, status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        try:
            if parsed.path == "/api/manifests":
                self.api_manifests()
            elif parsed.path == "/api/items":
                self.api_items(params)
            elif parsed.path == "/api/item":
                self.api_item(params)
            elif parsed.path == "/api/waveform":
                self.api_waveform(params)
            elif parsed.path == "/api/audio":
                self.api_audio(params)
            else:
                self.static_file(parsed.path)
        except FileNotFoundError as exc:
            self.send_error_json(str(exc), 404)
        except IndexError as exc:
            self.send_error_json(f"index out of range: {exc}", 404)
        except Exception as exc:  # noqa: BLE001
            self.send_error_json(f"{type(exc).__name__}: {exc}", 500)

    def api_manifests(self) -> None:
        self.send_json({"root": str(ROOT), "manifests": [m.as_dict(i) for i, m in enumerate(APP.manifests)]})

    def api_items(self, params: Dict[str, List[str]]) -> None:
        key = parse_int(params, "manifest")
        q = (params.get("q") or [""])[0].strip().lower()
        limit = max(1, min(1000, parse_int(params, "limit", 300)))
        manifest = APP.manifest(key)
        rows = manifest.summaries
        if q:
            rows = [
                row
                for row in rows
                if q in json.dumps(row, ensure_ascii=False).lower()
            ]
        self.send_json({"items": rows[:limit], "total": len(rows), "returned": min(limit, len(rows))})

    def api_item(self, params: Dict[str, List[str]]) -> None:
        key = parse_int(params, "manifest")
        manifest = APP.manifest(key)
        self.send_json(item_payload(key, manifest, parse_int(params, "index")))

    def api_audio(self, params: Dict[str, List[str]]) -> None:
        manifest = APP.manifest(parse_int(params, "manifest"))
        row = manifest.read_row(parse_int(params, "index"))
        audio_path = row_audio_path(row)
        self.send_file_with_range(audio_path, "audio/wav")

    def api_waveform(self, params: Dict[str, List[str]]) -> None:
        manifest = APP.manifest(parse_int(params, "manifest"))
        row = manifest.read_row(parse_int(params, "index"))
        audio_path = row_audio_path(row)
        self.send_json(waveform_payload(audio_path, parse_int(params, "width", 1600)))

    def static_file(self, request_path: str) -> None:
        if request_path in ("", "/"):
            request_path = "/index.html"
        rel = posixpath.normpath(unquote(request_path).lstrip("/"))
        path = (STATIC_DIR / rel).resolve()
        if not str(path).startswith(str(STATIC_DIR.resolve())) or not path.is_file():
            raise FileNotFoundError(request_path)
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_file_with_range(path, mime)

    def send_file_with_range(self, path: Path, mime: str) -> None:
        size = path.stat().st_size
        range_header = self.headers.get("Range")
        start, end = 0, size - 1
        status = HTTPStatus.OK
        if range_header:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header)
            if match:
                if match.group(1):
                    start = int(match.group(1))
                if match.group(2):
                    end = int(match.group(2))
                end = min(end, size - 1)
                status = HTTPStatus.PARTIAL_CONTENT
        if start < 0 or start >= size or end < start:
            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            return
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


APP: AppState


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a local duplex manifest/audio timeline viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--manifest", action="append", default=[], help="Manifest JSONL path. Can be passed multiple times.")
    parser.add_argument("--no_discover", action="store_true", help="Only load manifests passed with --manifest; do not scan outputs/**/manifest.jsonl.")
    args = parser.parse_args()

    manifests = discover_manifests(args.manifest, auto_discover=not args.no_discover)
    if not manifests:
        raise SystemExit("No manifest.jsonl files found under outputs/.")
    global APP
    APP = AppState(manifests)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Duplex viewer: http://{args.host}:{args.port}", flush=True)
    print("Loaded manifests:", flush=True)
    for i, manifest in enumerate(APP.manifests):
        print(f"  [{i}] {safe_rel(manifest.path)} ({len(manifest.offsets)} rows)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
