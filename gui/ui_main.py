# gui/ui_main.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
    QSpinBox, QDoubleSpinBox, QFileDialog, QListWidget, QTextEdit, QCheckBox
)


class MainUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("i2v_engine")

        root = QVBoxLayout(self)

        # prompt
        root.addWidget(QLabel("Prompt"))
        self.prompt = QLineEdit()
        root.addWidget(self.prompt)

        root.addWidget(QLabel("Negative prompt"))
        self.neg = QLineEdit()
        root.addWidget(self.neg)

        # settings
        row = QHBoxLayout()
        self.fps = QSpinBox(); self.fps.setRange(1, 60); self.fps.setValue(24)
        self.seconds = QSpinBox(); self.seconds.setRange(1, 30); self.seconds.setValue(8)
        self.steps = QSpinBox(); self.steps.setRange(5, 100); self.steps.setValue(30)
        self.guidance = QDoubleSpinBox(); self.guidance.setRange(1.0, 20.0); self.guidance.setValue(7.5)
        row.addWidget(QLabel("FPS")); row.addWidget(self.fps)
        row.addWidget(QLabel("Seconds")); row.addWidget(self.seconds)
        row.addWidget(QLabel("Steps")); row.addWidget(self.steps)
        row.addWidget(QLabel("Guidance")); row.addWidget(self.guidance)
        root.addLayout(row)

        # toggles
        row2 = QHBoxLayout()
        self.temporal = QCheckBox("Temporal face-lock"); self.temporal.setChecked(True)
        self.restore = QCheckBox("Face restore"); self.restore.setChecked(False)
        row2.addWidget(self.temporal)
        row2.addWidget(self.restore)
        root.addLayout(row2)

        # buttons
        btns = QHBoxLayout()
        self.btn_install = QPushButton("Install deps (1 click)")
        self.btn_update = QPushButton("Update from GitHub")
        self.btn_generate = QPushButton("Generate")
        btns.addWidget(self.btn_install)
        btns.addWidget(self.btn_update)
        btns.addWidget(self.btn_generate)
        root.addLayout(btns)

        # status
        self.status = QLabel("Idle")
        root.addWidget(self.status)

        # history list
        root.addWidget(QLabel("History"))
        self.history = QListWidget()
        root.addWidget(self.history)

        # logs
        root.addWidget(QLabel("Logs"))
        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        root.addWidget(self.logs)