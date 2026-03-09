from __future__ import annotations

from pathlib import Path

from models.settings import AppSettings


class SessionService:
    def __init__(self, session_path: str):
        self.path = Path(session_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, settings: AppSettings) -> None:
        self.path.write_text(settings.model_dump_json(indent=2), encoding="utf-8")

    def load(self) -> AppSettings:
        if not self.path.exists():
            return AppSettings()
        return AppSettings.model_validate_json(self.path.read_text(encoding="utf-8"))
