from __future__ import annotations

from pathlib import Path
from typing import List

from models.settings import LoraItem


class LoraManagerService:
    def __init__(self, library_dir: str):
        self.library_dir = Path(library_dir)
        self.library_dir.mkdir(parents=True, exist_ok=True)

    def scan(self) -> List[LoraItem]:
        items: List[LoraItem] = []
        for path in sorted(self.library_dir.glob("*.safetensors")):
            items.append(LoraItem(name=path.stem, path=str(path), scale=0.75, enabled=False))
        return items
