from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import List

from models.history import HistoryDB, HistoryItem


class HistoryService:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "history.json"

    def load(self) -> HistoryDB:
        if not self.path.exists():
            return HistoryDB()
        return HistoryDB(**json.loads(self.path.read_text(encoding="utf-8")))

    def save(self, db: HistoryDB) -> None:
        self.path.write_text(db.model_dump_json(indent=2), encoding="utf-8")

    def append(self, item: HistoryItem) -> None:
        db = self.load()
        db.items.insert(0, item)
        self.save(db)

    def new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def list_items(self) -> List[HistoryItem]:
        return self.load().items
