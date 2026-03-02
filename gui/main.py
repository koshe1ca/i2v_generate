# gui/main.py
import json
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QThread

from gui.ui_main import MainUI
from gui.worker import Worker

from models.settings import EngineSettings, LoRAItem
from services.install_service import InstallService
from services.update_service import UpdateService
from services.pipeline_service import PipelineService


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_history(outputs_dir: str):
    p = Path(outputs_dir) / "history.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("items", [])


def main():
    app = QApplication([])
    ui = MainUI()

    # defaults
    ui.prompt.setText("realistic portrait, cinematic lighting, sharp, high detail")
    ui.neg.setText("low quality, blurry, artifacts, deformed, extra fingers")

    def refresh_history(outputs_dir="outputs"):
        ui.history.clear()
        items = load_history(outputs_dir)
        for it in items[:200]:
            ui.history.addItem(f"{it['created_at']} | {it['mode']} | {Path(it['output_video']).name}")

    refresh_history()

    def run_in_thread(fn, on_ok):
        thread = QThread()
        worker = Worker(fn)
        worker.moveToThread(thread)

        worker.status.connect(lambda s: ui.status.setText(s))
        worker.failed.connect(lambda e: ui.logs.append(e))
        worker.finished.connect(lambda res: on_ok(res))
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)

        thread.started.connect(worker.run)
        thread.start()
        return thread

    # INSTALL
    def on_install():
        def task(w):
            w.status.emit("Installing requirements...")
            return InstallService().install_requirements(str(PROJECT_ROOT / "requirements.txt"))

        def ok(res):
            ui.status.setText(f"Install: {res}")
            ui.logs.append(f"Install result: {res}")

        run_in_thread(task, ok)

    # UPDATE
    def on_update():
        def task(w):
            w.status.emit("Updating from GitHub...")
            s = EngineSettings()
            return UpdateService(s.github_owner, s.github_repo, s.github_branch).update_from_github_zip(str(PROJECT_ROOT))

        def ok(res):
            ui.status.setText(f"Update: {res}")
            ui.logs.append(f"Update result: {res}")

        run_in_thread(task, ok)

    # GENERATE
    def on_generate():
        def task(w):
            w.status.emit("Loading pipeline...")
            s = EngineSettings()
            s.prompt = ui.prompt.text().strip()
            s.negative_prompt = ui.neg.text().strip()
            s.video.fps = int(ui.fps.value())
            s.video.seconds = int(ui.seconds.value())
            s.video.steps = int(ui.steps.value())
            s.video.guidance = float(ui.guidance.value())
            s.temporal.enable = ui.temporal.isChecked()
            s.face_restore.enable = ui.restore.isChecked()

            # TODO: сюда позже добавим GUI-поля для LoRA list
            # пример:
            # s.lora.items = [LoRAItem(path="loras/face.safetensors", scale=1.0)]

            pipe = PipelineService(s)
            pipe.load()

            w.status.emit("Generating...")
            mp4 = pipe.generate_ad(pose_dir=s.pose_dir)
            return str(mp4)

        def ok(res):
            ui.status.setText(f"Done: {res}")
            ui.logs.append(f"Generated: {res}")
            refresh_history()

        run_in_thread(task, ok)

    ui.btn_install.clicked.connect(on_install)
    ui.btn_update.clicked.connect(on_update)
    ui.btn_generate.clicked.connect(on_generate)

    ui.resize(1000, 700)
    ui.show()
    app.exec()


if __name__ == "__main__":
    main()