from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QObject, QThread, QTimer, Signal

from models.settings import AppSettings
from services.history_service import HistoryService
from services.pipeline_service import PipelineService
from services.session_service import SessionService
from services.system_monitor_service import SystemMonitorService


class GenerateWorker(QObject):
    progress = Signal(str, int)
    preview_ready = Signal(str, str, str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, settings: AppSettings):
        super().__init__()
        self.settings = settings
        self.pipeline = PipelineService(settings)

    def run(self):
        try:
            path = self.pipeline.generate(stage_cb=self._stage, preview_cb=self._preview)
            self.finished_ok.emit(str(path))
        except Exception as exc:
            self.failed.emit(str(exc))

    def cancel(self):
        self.pipeline.cancel()

    def _stage(self, stage: str, step: int):
        self.progress.emit(stage, step)

    def _preview(self, first, middle, last):
        import tempfile
        from pathlib import Path
        tmp = Path(tempfile.gettempdir()) / "i2v_gui_previews"
        tmp.mkdir(parents=True, exist_ok=True)
        p1, p2, p3 = tmp / "first.png", tmp / "middle.png", tmp / "last.png"
        first.save(p1)
        middle.save(p2)
        last.save(p3)
        self.preview_ready.emit(str(p1), str(p2), str(p3))


class GenerateController(QObject):
    monitor = Signal(dict)
    progress = Signal(str, int)
    preview_ready = Signal(str, str, str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, session_path: str, output_dir: str):
        super().__init__()
        self.session = SessionService(session_path)
        self.history = HistoryService(output_dir)
        self.monitor_service = SystemMonitorService()
        self.thread: Optional[QThread] = None
        self.worker: Optional[GenerateWorker] = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick_monitor)

    def load_settings(self) -> AppSettings:
        return self.session.load()

    def save_settings(self, settings: AppSettings) -> None:
        self.session.save(settings)

    def start(self, settings: AppSettings) -> None:
        self.save_settings(settings)
        self.thread = QThread()
        self.worker = GenerateWorker(settings)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress)
        self.worker.preview_ready.connect(self.preview_ready)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished_ok.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.timer.start(1000)
        self.thread.start()

    def cancel(self) -> None:
        if self.worker:
            self.worker.cancel()

    def _tick_monitor(self):
        self.monitor.emit(self.monitor_service.snapshot())

    def _on_finished(self, path: str):
        self.timer.stop()
        self.finished_ok.emit(path)

    def _on_failed(self, message: str):
        self.timer.stop()
        self.failed.emit(message)
