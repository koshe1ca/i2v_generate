from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QPlainTextEdit,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from controllers.generate_controller import GenerateController
from models.history import HistoryItem
from models.settings import AppSettings, LoraItem
from services.lora_manager_service import LoraManagerService
from services.model_manager_service import ModelManagerService


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("I2V Generate Desktop")
        self.resize(1500, 950)

        self.app_dir = Path.home() / ".i2v_generate_app"
        self.output_dir = self.app_dir / "outputs"
        self.models_dir = self.app_dir / "models"
        self.lora_dir = self.models_dir / "lora"
        self.base_models_dir = self.models_dir / "base"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lora_dir.mkdir(parents=True, exist_ok=True)
        self.base_models_dir.mkdir(parents=True, exist_ok=True)

        self.controller = GenerateController(
            session_path=str(self.app_dir / "last_session.json"),
            output_dir=str(self.output_dir),
        )
        self.settings = self.controller.load_settings()
        if not self.settings.output_override_dir:
            self.settings.output_override_dir = str(self.output_dir)

        self.lora_manager = LoraManagerService(str(self.lora_dir))
        self.model_manager = ModelManagerService(str(self.base_models_dir))

        self._build_ui()
        self._bind_controller()
        self._load_settings_into_ui()
        self.refresh_lora_library()
        self.refresh_model_library()
        self.refresh_history()

    def _build_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.generate_tab = QWidget()
        self.history_tab = QWidget()
        self.models_tab = QWidget()
        self.logs_tab = QWidget()
        self.tabs.addTab(self.generate_tab, "Generate")
        self.tabs.addTab(self.history_tab, "History")
        self.tabs.addTab(self.models_tab, "Models & LoRA")
        self.tabs.addTab(self.logs_tab, "Logs")

        self._build_generate_tab()
        self._build_history_tab()
        self._build_models_tab()
        self._build_logs_tab()

    def _build_generate_tab(self):
        layout = QHBoxLayout(self.generate_tab)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        form = QFormLayout()

        self.mode_box = QComboBox(); self.mode_box.addItems(["photo_prompt", "photo_plus_video_motion"])
        self.base_model_edit = QLineEdit()
        self.motion_adapter_edit = QLineEdit()
        self.prompt_edit = QTextEdit(); self.prompt_edit.setFixedHeight(110)
        self.negative_prompt_edit = QTextEdit(); self.negative_prompt_edit.setFixedHeight(70)
        self.input_image_edit = QLineEdit(); self.input_image_btn = QPushButton("Browse")
        self.ref_video_edit = QLineEdit(); self.ref_video_btn = QPushButton("Browse")
        self.output_dir_edit = QLineEdit(); self.output_dir_btn = QPushButton("Browse")
        self.preset_box = QComboBox(); self.preset_box.addItems(["fast", "balanced", "high"])

        row_img = QWidget(); row_img_l = QHBoxLayout(row_img); row_img_l.setContentsMargins(0, 0, 0, 0); row_img_l.addWidget(self.input_image_edit); row_img_l.addWidget(self.input_image_btn)
        row_vid = QWidget(); row_vid_l = QHBoxLayout(row_vid); row_vid_l.setContentsMargins(0, 0, 0, 0); row_vid_l.addWidget(self.ref_video_edit); row_vid_l.addWidget(self.ref_video_btn)
        row_out = QWidget(); row_out_l = QHBoxLayout(row_out); row_out_l.setContentsMargins(0, 0, 0, 0); row_out_l.addWidget(self.output_dir_edit); row_out_l.addWidget(self.output_dir_btn)

        form.addRow("Mode", self.mode_box)
        form.addRow("Quality preset", self.preset_box)
        form.addRow("Base model", self.base_model_edit)
        form.addRow("Motion adapter", self.motion_adapter_edit)
        form.addRow("Prompt", self.prompt_edit)
        form.addRow("Negative", self.negative_prompt_edit)
        form.addRow("Photo", row_img)
        form.addRow("Reference video", row_vid)
        form.addRow("Output folder", row_out)

        adv = QGroupBox("Advanced settings")
        adv_form = QFormLayout(adv)
        self.frames_spin = QSpinBox(); self.frames_spin.setRange(4, 256)
        self.steps_spin = QSpinBox(); self.steps_spin.setRange(1, 200)
        self.guidance_spin = QDoubleSpinBox(); self.guidance_spin.setRange(0.0, 20.0); self.guidance_spin.setSingleStep(0.1)
        self.width_spin = QSpinBox(); self.width_spin.setRange(256, 2048); self.width_spin.setSingleStep(64)
        self.height_spin = QSpinBox(); self.height_spin.setRange(256, 2048); self.height_spin.setSingleStep(64)
        self.fps_spin = QSpinBox(); self.fps_spin.setRange(1, 120)
        self.seed_spin = QSpinBox(); self.seed_spin.setRange(0, 2_147_483_647)
        self.temporal_cb = QCheckBox("Enable temporal stabilization")
        self.face_restore_cb = QCheckBox("Enable face restore")
        self.controlnet_cb = QCheckBox("Enable pose control")
        self.save_frames_cb = QCheckBox("Save frames")
        self.face_lock_cb = QCheckBox("Face lock only")
        self.temporal_strength = QDoubleSpinBox(); self.temporal_strength.setRange(0, 1); self.temporal_strength.setSingleStep(0.05)
        self.control_strength = QDoubleSpinBox(); self.control_strength.setRange(0, 2); self.control_strength.setSingleStep(0.05)
        self.ip_adapter_scale = QDoubleSpinBox(); self.ip_adapter_scale.setRange(0, 2); self.ip_adapter_scale.setSingleStep(0.05)
        self.long_video_cb = QCheckBox("Long video mode")
        self.target_duration_spin = QDoubleSpinBox(); self.target_duration_spin.setRange(1, 60); self.target_duration_spin.setSingleStep(1)
        self.chunk_frames_spin = QSpinBox(); self.chunk_frames_spin.setRange(4, 64)
        self.overlap_frames_spin = QSpinBox(); self.overlap_frames_spin.setRange(0, 32)
        self.target_fps_after_rife_spin = QSpinBox(); self.target_fps_after_rife_spin.setRange(1, 120)

        adv_form.addRow("Frames / chunk", self.frames_spin)
        adv_form.addRow("Steps", self.steps_spin)
        adv_form.addRow("Guidance", self.guidance_spin)
        adv_form.addRow("Width", self.width_spin)
        adv_form.addRow("Height", self.height_spin)
        adv_form.addRow("FPS", self.fps_spin)
        adv_form.addRow("Seed", self.seed_spin)
        adv_form.addRow("IP-Adapter scale", self.ip_adapter_scale)
        adv_form.addRow(self.temporal_cb)
        adv_form.addRow(self.face_lock_cb)
        adv_form.addRow("Temporal strength", self.temporal_strength)
        adv_form.addRow(self.face_restore_cb)
        adv_form.addRow(self.controlnet_cb)
        adv_form.addRow("Control strength", self.control_strength)
        adv_form.addRow(self.save_frames_cb)
        adv_form.addRow(self.long_video_cb)
        adv_form.addRow("Target duration (sec)", self.target_duration_spin)
        adv_form.addRow("Chunk frames", self.chunk_frames_spin)
        adv_form.addRow("Overlap frames", self.overlap_frames_spin)
        adv_form.addRow("Target FPS after RIFE", self.target_fps_after_rife_spin)

        left_layout.addLayout(form)
        left_layout.addWidget(adv)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("Start generation")
        self.cancel_btn = QPushButton("Cancel generation")
        self.cancel_btn.setEnabled(False)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.cancel_btn)
        left_layout.addLayout(btn_row)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.stage_label = QLabel("Idle")
        self.progress_bar = QProgressBar(); self.progress_bar.setRange(0, 100)
        self.monitor_label = QLabel("CPU: -, RAM: -, GPU: -")
        right_layout.addWidget(self.stage_label)
        right_layout.addWidget(self.progress_bar)
        right_layout.addWidget(self.monitor_label)

        prev_group = QGroupBox("Preview frames")
        prev_layout = QGridLayout(prev_group)
        self.preview_first = QLabel("First"); self.preview_middle = QLabel("Middle"); self.preview_last = QLabel("Last")
        for lbl in [self.preview_first, self.preview_middle, self.preview_last]:
            lbl.setFixedSize(320, 240)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("border:1px solid #555; background:#222;")
        prev_layout.addWidget(QLabel("First"), 0, 0)
        prev_layout.addWidget(QLabel("Middle"), 0, 1)
        prev_layout.addWidget(QLabel("Last"), 0, 2)
        prev_layout.addWidget(self.preview_first, 1, 0)
        prev_layout.addWidget(self.preview_middle, 1, 1)
        prev_layout.addWidget(self.preview_last, 1, 2)
        right_layout.addWidget(prev_group)

        self.selected_loras = QListWidget()
        right_layout.addWidget(QLabel("Active LoRAs"))
        right_layout.addWidget(self.selected_loras)

        layout.addWidget(left, 2)
        layout.addWidget(right, 3)

        self.input_image_btn.clicked.connect(lambda: self._pick_file(self.input_image_edit, "Image (*.png *.jpg *.jpeg *.webp)"))
        self.ref_video_btn.clicked.connect(lambda: self._pick_file(self.ref_video_edit, "Video (*.mp4 *.mov *.mkv *.avi)"))
        self.output_dir_btn.clicked.connect(self._pick_output_dir)
        self.start_btn.clicked.connect(self.start_generation)
        self.cancel_btn.clicked.connect(self.controller.cancel)
        self.preset_box.currentTextChanged.connect(self._preset_changed)

    def _build_history_tab(self):
        layout = QVBoxLayout(self.history_tab)
        self.history_list = QListWidget()
        self.repeat_btn = QPushButton("Repeat generation from History")
        self.open_video_btn = QPushButton("Open selected video")
        row = QHBoxLayout(); row.addWidget(self.repeat_btn); row.addWidget(self.open_video_btn)
        layout.addLayout(row)
        layout.addWidget(self.history_list)
        self.repeat_btn.clicked.connect(self.repeat_from_history)
        self.open_video_btn.clicked.connect(self.open_selected_history_video)

    def _build_models_tab(self):
        layout = QHBoxLayout(self.models_tab)
        left = QVBoxLayout(); right = QVBoxLayout()
        self.models_list = QListWidget(); self.refresh_models_btn = QPushButton("Refresh models"); self.add_model_btn = QPushButton("Add base model file/folder")
        self.lora_list = QListWidget(); self.refresh_lora_btn = QPushButton("Refresh LoRA library"); self.add_lora_btn = QPushButton("Add LoRA file")
        self.use_lora_btn = QPushButton("Activate selected LoRA")
        self.remove_lora_btn = QPushButton("Remove selected active LoRA")
        for w in [QLabel("Model manager"), self.models_list, self.refresh_models_btn, self.add_model_btn]: left.addWidget(w)
        for w in [QLabel("LoRA manager"), self.lora_list, self.refresh_lora_btn, self.add_lora_btn, self.use_lora_btn, self.remove_lora_btn]: right.addWidget(w)
        wrap_l = QWidget(); wrap_l.setLayout(left); wrap_r = QWidget(); wrap_r.setLayout(right)
        layout.addWidget(wrap_l); layout.addWidget(wrap_r)
        self.refresh_models_btn.clicked.connect(self.refresh_model_library)
        self.refresh_lora_btn.clicked.connect(self.refresh_lora_library)
        self.add_lora_btn.clicked.connect(self.add_lora_file)
        self.use_lora_btn.clicked.connect(self.activate_selected_lora)
        self.remove_lora_btn.clicked.connect(self.remove_active_lora)
        self.add_model_btn.clicked.connect(self.add_model_path)

    def _build_logs_tab(self):
        layout = QVBoxLayout(self.logs_tab)
        self.logs_view = QPlainTextEdit(); self.logs_view.setReadOnly(True)
        self.refresh_logs_btn = QPushButton("Refresh logs")
        layout.addWidget(self.refresh_logs_btn)
        layout.addWidget(self.logs_view)
        self.refresh_logs_btn.clicked.connect(self.refresh_logs)

    def _bind_controller(self):
        self.controller.progress.connect(self._on_progress)
        self.controller.monitor.connect(self._on_monitor)
        self.controller.preview_ready.connect(self._on_preview)
        self.controller.finished_ok.connect(self._on_finished)
        self.controller.failed.connect(self._on_failed)

    def _load_settings_into_ui(self):
        s = self.settings
        self.mode_box.setCurrentText(s.mode)
        self.preset_box.setCurrentText(s.quality_preset)
        self.base_model_edit.setText(s.base_model)
        self.motion_adapter_edit.setText(s.motion_adapter)
        self.prompt_edit.setPlainText(s.prompt)
        self.negative_prompt_edit.setPlainText(s.negative_prompt)
        self.input_image_edit.setText(s.input_image or "")
        self.ref_video_edit.setText(s.ref_video or "")
        self.output_dir_edit.setText(s.output_override_dir or str(self.output_dir))
        self.frames_spin.setValue(s.video.num_frames)
        self.steps_spin.setValue(s.video.steps)
        self.guidance_spin.setValue(s.video.guidance)
        self.width_spin.setValue(s.video.width)
        self.height_spin.setValue(s.video.height)
        self.fps_spin.setValue(s.video.fps)
        self.seed_spin.setValue(s.video.seed)
        self.ip_adapter_scale.setValue(s.ip_adapter.scale)
        self.temporal_cb.setChecked(s.temporal.enable)
        self.face_restore_cb.setChecked(s.face_restore.enable)
        self.controlnet_cb.setChecked(s.controlnet.enable)
        self.save_frames_cb.setChecked(s.video.save_frames)
        self.face_lock_cb.setChecked(s.temporal.face_lock_only)
        self.temporal_strength.setValue(s.temporal.strength)
        self.control_strength.setValue(s.controlnet.conditioning_scale)
        self.long_video_cb.setChecked(s.long_video.enabled)
        self.target_duration_spin.setValue(s.long_video.target_duration_sec)
        self.chunk_frames_spin.setValue(s.long_video.chunk_frames)
        self.overlap_frames_spin.setValue(s.long_video.overlap_frames)
        self.target_fps_after_rife_spin.setValue(s.rife.target_fps)
        self._refresh_active_lora_widget()
        self.refresh_logs()

    def _collect_settings(self) -> AppSettings:
        s = self.settings
        s.mode = self.mode_box.currentText()
        s.base_model = self.base_model_edit.text().strip()
        s.motion_adapter = self.motion_adapter_edit.text().strip()
        s.prompt = self.prompt_edit.toPlainText().strip()
        s.negative_prompt = self.negative_prompt_edit.toPlainText().strip()
        s.input_image = self.input_image_edit.text().strip() or None
        s.ref_video = self.ref_video_edit.text().strip() or None
        s.output_override_dir = self.output_dir_edit.text().strip() or str(self.output_dir)
        s.video.num_frames = self.frames_spin.value()
        s.video.steps = self.steps_spin.value()
        s.video.guidance = self.guidance_spin.value()
        s.video.width = self.width_spin.value()
        s.video.height = self.height_spin.value()
        s.video.fps = self.fps_spin.value()
        s.video.seed = self.seed_spin.value()
        s.ip_adapter.scale = self.ip_adapter_scale.value()
        s.temporal.enable = self.temporal_cb.isChecked()
        s.temporal.face_lock_only = self.face_lock_cb.isChecked()
        s.temporal.strength = self.temporal_strength.value()
        s.face_restore.enable = self.face_restore_cb.isChecked()
        s.controlnet.enable = self.controlnet_cb.isChecked()
        s.controlnet.conditioning_scale = self.control_strength.value()
        s.video.save_frames = self.save_frames_cb.isChecked()
        s.quality_preset = self.preset_box.currentText()
        s.long_video.enabled = self.long_video_cb.isChecked()
        s.long_video.target_duration_sec = self.target_duration_spin.value()
        s.long_video.chunk_frames = self.chunk_frames_spin.value()
        s.long_video.overlap_frames = self.overlap_frames_spin.value()
        s.rife.target_fps = self.target_fps_after_rife_spin.value()
        return s

    def start_generation(self):
        self.settings = self._collect_settings()
        self.stage_label.setText("Starting...")
        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.controller.start(self.settings)

    def repeat_from_history(self):
        item = self._selected_history_item()
        if not item:
            return
        self.settings = AppSettings.model_validate(item.settings)
        self._load_settings_into_ui()
        self.tabs.setCurrentWidget(self.generate_tab)

    def open_selected_history_video(self):
        item = self._selected_history_item()
        if item:
            QDesktopServices.openUrl(QUrl.fromLocalFile(item.output_video))

    def refresh_history(self):
        self.history_list.clear()
        for item in self.controller.history.list_items():
            it = QListWidgetItem(f"{item.created_at} | {item.mode} | {Path(item.output_video).name}")
            it.setData(Qt.UserRole, item)
            self.history_list.addItem(it)

    def refresh_logs(self):
        log_path = self.output_dir / "errors.log"
        self.logs_view.setPlainText(log_path.read_text(encoding="utf-8") if log_path.exists() else "No errors yet.")

    def refresh_lora_library(self):
        self.lora_list.clear()
        for item in self.lora_manager.scan():
            row = QListWidgetItem(f"{item.name} | {Path(item.path).name}")
            row.setData(Qt.UserRole, item)
            self.lora_list.addItem(row)
        self._refresh_active_lora_widget()

    def refresh_model_library(self):
        self.models_list.clear()
        for path in self.model_manager.scan():
            self.models_list.addItem(path)

    def add_lora_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose LoRA", str(Path.home()), "LoRA (*.safetensors)")
        if not path:
            return
        target = self.lora_dir / Path(path).name
        if Path(path) != target:
            target.write_bytes(Path(path).read_bytes())
        self.refresh_lora_library()

    def add_model_path(self):
        path = QFileDialog.getExistingDirectory(self, "Choose diffusers model folder", str(Path.home()))
        if path:
            self.base_model_edit.setText(path)
            self.refresh_model_library()

    def activate_selected_lora(self):
        item = self.lora_list.currentItem()
        if not item:
            return
        lora: LoraItem = item.data(Qt.UserRole)
        if any(x.path == lora.path for x in self.settings.lora.items):
            return
        lora.enabled = True
        self.settings.lora.items.append(lora)
        self._refresh_active_lora_widget()

    def remove_active_lora(self):
        row = self.selected_loras.currentRow()
        if row >= 0:
            del self.settings.lora.items[row]
            self._refresh_active_lora_widget()

    def _refresh_active_lora_widget(self):
        self.selected_loras.clear()
        for item in self.settings.lora.items:
            self.selected_loras.addItem(f"{item.name} | scale={item.scale:.2f} | {Path(item.path).name}")

    def _pick_file(self, target: QLineEdit, filt: str):
        path, _ = QFileDialog.getOpenFileName(self, "Choose file", str(Path.home()), filt)
        if path:
            target.setText(path)

    def _pick_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Choose output directory", self.output_dir_edit.text() or str(Path.home()))
        if path:
            self.output_dir_edit.setText(path)

    def _on_progress(self, stage: str, value: int):
        self.stage_label.setText(stage)
        if stage.startswith("Chunk "):
            # value is 0 for chunk transitions; let label do the work
            return
        if stage == "Generating" and self.steps_spin.value() > 0:
            self.progress_bar.setValue(min(100, int((value / self.steps_spin.value()) * 100)))
        elif stage == "Done":
            self.progress_bar.setValue(100)

    def _on_monitor(self, snap: dict):
        self.monitor_label.setText(
            f"CPU {snap['cpu_percent']}% | RAM {snap['ram_percent']}% ({snap['ram_used_gb']}/{snap['ram_total_gb']} GB) | "
            f"GPU {snap['gpu_name']} {snap['gpu_percent']}% | VRAM {snap['gpu_mem_used_mb']}/{snap['gpu_mem_total_mb']} MB | {snap['gpu_temp_c']}°C"
        )

    def _set_preview(self, label: QLabel, path: str):
        pix = QPixmap(path)
        if pix.isNull():
            label.setText("No preview")
            return
        label.setPixmap(pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _on_preview(self, p1: str, p2: str, p3: str):
        self._set_preview(self.preview_first, p1)
        self._set_preview(self.preview_middle, p2)
        self._set_preview(self.preview_last, p3)

    def _on_finished(self, path: str):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.refresh_history()
        self.refresh_logs()
        QMessageBox.information(self, "Done", f"Video saved:\n{path}")

    def _on_failed(self, message: str):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.refresh_logs()
        QMessageBox.critical(self, "Generation failed", message)

    def _selected_history_item(self) -> Optional[HistoryItem]:
        item = self.history_list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def _preset_changed(self, preset: str):
        self.settings.apply_preset(preset)
        self._load_settings_into_ui()
