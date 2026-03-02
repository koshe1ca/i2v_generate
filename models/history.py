# models/history.py
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime


class HistoryItem(BaseModel):
    created_at: str
    mode: str  # "ad" | "svd" | "ad_pose"
    prompt: str
    negative_prompt: str
    input_image: Optional[str] = None
    ref_video: Optional[str] = None
    pose_dir: Optional[str] = None
    output_video: Optional[str] = None
    output_frames_dir: Optional[str] = None
    settings: Dict = {}


class HistoryDB(BaseModel):
    items: List[HistoryItem] = []

    @staticmethod
    def now_iso():
        return datetime.now().isoformat(timespec="seconds")