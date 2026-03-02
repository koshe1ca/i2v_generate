# gui/worker.py
from PySide6.QtCore import QObject, Signal, Slot, QThread
import traceback


class Worker(QObject):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    @Slot()
    def run(self):
        try:
            self.status.emit("Started...")
            result = self.fn(self)
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())