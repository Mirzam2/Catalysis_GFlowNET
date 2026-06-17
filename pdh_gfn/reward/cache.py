"""Кэш наград (уровень 2 канонизации) с персистентностью в JSONL."""
import json
from pathlib import Path
from typing import Dict, Optional

from .gates import RewardBreakdown


class RewardCache:
    def __init__(self, path: Optional[Path] = None):
        self._mem: Dict[str, RewardBreakdown] = {}
        self._path = Path(path) if path else None
        if self._path and self._path.exists():
            for line in self._path.read_text().splitlines():
                rec = json.loads(line)
                key = rec.pop("key")
                self._mem[key] = RewardBreakdown(**rec)

    def get(self, key: str) -> Optional[RewardBreakdown]:
        return self._mem.get(key)

    def put(self, key: str, value: RewardBreakdown) -> None:
        self._mem[key] = value
        if self._path:
            with self._path.open("a") as f:
                f.write(json.dumps({"key": key, **value.__dict__}) + "\n")

    def __len__(self) -> int:
        return len(self._mem)

    @property
    def hit_stats(self) -> int:
        return len(self._mem)
