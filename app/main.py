from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from gui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("I2V Generate Desktop")
    app.setStyleSheet(
        """
        QWidget { background: #1e1f24; color: #f1f3f5; font-size: 13px; }
        QPushButton { background: #3b82f6; border: none; padding: 8px 12px; border-radius: 8px; }
        QPushButton:disabled { background: #444; }
        QLineEdit, QTextEdit, QPlainTextEdit, QListWidget, QComboBox, QSpinBox, QDoubleSpinBox {
            background: #262932; border: 1px solid #3b4252; border-radius: 8px; padding: 6px;
        }
        QGroupBox { border: 1px solid #3b4252; border-radius: 10px; margin-top: 12px; padding-top: 10px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
        QTabWidget::pane { border: 1px solid #3b4252; border-radius: 10px; }
        QProgressBar { border: 1px solid #3b4252; border-radius: 8px; text-align: center; background: #262932; }
        QProgressBar::chunk { background: #10b981; border-radius: 8px; }
        """
    )
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
