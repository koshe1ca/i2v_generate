from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HistoryItem(BaseModel):
    id: str
    created_at: str
    mode: str
    prompt: str
    negative_prompt: str = ""
    input_image: Optional[str] = None
    ref_video: Optional[str] = None
    output_video: str
    output_frames_dir: Optional[str] = None
    loras: List[Dict[str, Any]] = Field(default_factory=list)
    preview_first: Optional[str] = None
    preview_middle: Optional[str] = None
    preview_last: Optional[str] = None
    settings: Dict[str, Any] = Field(default_factory=dict)


class HistoryDB(BaseModel):
    items: List[HistoryItem] = Field(default_factory=list)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    @staticmethod
    def file_path(output_dir: str | Path) -> Path:
        return Path(output_dir) / "history.json"
