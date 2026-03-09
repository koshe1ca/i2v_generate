from __future__ import annotations

from pathlib import Path
from typing import List


class ModelManagerService:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def scan(self) -> List[str]:
        found: List[str] = []
        for p in sorted(self.model_dir.iterdir()):
            if p.is_dir() or p.suffix in {".ckpt", ".safetensors"}:
                found.append(str(p))
        return found
