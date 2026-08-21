from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .io import atomic_write_json, read_json


class RunState:
    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / "run_state.json"
        self.data: Dict[str, Any] = read_json(self.path) if self.path.exists() else {"stages": {}}

    def update(self, stage: str, status: str, **details: Any) -> None:
        now = datetime.now(timezone.utc).isoformat()
        current = dict(self.data.setdefault("stages", {}).get(stage) or {})
        if status == "running" and "started_at" not in current:
            current["started_at"] = now
        current.update(details)
        current["status"] = status
        current["updated_at"] = now
        self.data["stages"][stage] = current
        atomic_write_json(self.path, self.data)

    def is_complete(self, stage: str) -> bool:
        return self.data.get("stages", {}).get(stage, {}).get("status") == "complete"
