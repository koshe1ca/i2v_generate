from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Job:
    # что генерим
    mode: str  # "svd" | "ad"

    # входные данные
    image_path: Optional[str] = None     # для svd
    prompt: Optional[str] = None         # для ad
    negative_prompt: Optional[str] = None

    # управление длительностью
    seconds: float = 2.0                 # для ad: total_frames = seconds * fps

    # выход
    out_name: str = "result"

    # будущее: реф-видео
    ref_video_path: Optional[str] = None