from __future__ import annotations

import traceback
from datetime import datetime
from pathlib import Path


class ErrorLogService:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "errors.log"

    def log_exception(self, exc: BaseException, context: str = "") -> str:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = (
            f"[{stamp}] {context}\n"
            f"{type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()}\n"
            f"{'-' * 80}\n"
        )
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(text)
        return str(self.log_path)
