from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


@dataclass(frozen=True)
class SetupConfig:
    project_root: Path
    python_version: str = "3.10"
    venv_dir: str = ".venv"
    requirements_file: str = "requirements.txt"

    # ---- installers (winget ids) ----
    winget_git_id: str = "Git.Git"
    winget_buildtools_id: str = "Microsoft.VisualStudio.2022.BuildTools"
    winget_vcpp_id: str = "Microsoft.VCRedist.2015+.x64"

    # python via winget
    winget_python_id: str = "Python.Python.3.10"


class InstallService:
    """
    Windows-only: one-command setup (Python, venv, deps).
    GUI button later will call the same entry.
    """

    def __init__(self, cfg: SetupConfig):
        self.cfg = cfg

    # -----------------------
    # Public API
    # -----------------------
    def setup_all(self) -> None:
        self._assert_windows()

        console.rule("[bold cyan]SETUP START[/bold cyan]")

        self._ensure_winget()

        # 1) Python
        self._ensure_python()

        # 2) Git (for updates later)
        self._ensure_winget_package(self.cfg.winget_git_id, "Git")

        # 3) VC++ Runtime (fix many dll problems)
        self._ensure_winget_package(self.cfg.winget_vcpp_id, "VC++ Runtime")

        # 4) Build Tools (only if something needs compilation)
        self._ensure_winget_package(self.cfg.winget_buildtools_id, "VS Build Tools 2022")

        # 5) venv
        venv_python = self._ensure_venv()

        # 6) pip deps
        self._pip_install_requirements(venv_python)

        # 7) quick checks
        self._quick_check(venv_python)

        console.rule("[bold green]SETUP DONE[/bold green]")

    # -----------------------
    # Internals
    # -----------------------
    def _assert_windows(self):
        if os.name != "nt":
            raise RuntimeError("Setup service is Windows-only for now.")

    def _run(self, cmd: list[str], *, check: bool = True, shell: bool = False) -> subprocess.CompletedProcess:
        console.print(f"[dim]$ {' '.join(cmd) if isinstance(cmd, list) else cmd}[/dim]")
        return subprocess.run(cmd, check=check, shell=shell)

    def _ensure_winget(self):
        if shutil.which("winget") is None:
            raise RuntimeError(
                "winget not found. Install 'App Installer' from Microsoft Store, then retry."
            )

    def _ensure_winget_package(self, pkg_id: str, label: str) -> None:
        # If already installed, winget will just say it's installed / up to date.
        console.print(f"[cyan]Ensuring:[/cyan] {label}")
        self._run(["winget", "install", "-e", "--id", pkg_id])

    def _ensure_python(self) -> None:
        # If current python is 3.10 x64, we can skip installing
        cur = f"{sys.version_info.major}.{sys.version_info.minor}"
        if cur == self.cfg.python_version:
            console.print(f"[green]Python {cur} already running[/green]")
            return

        console.print(f"[yellow]Current python is {cur}. Ensuring Python {self.cfg.python_version} via winget...[/yellow]")
        self._ensure_winget_package(self.cfg.winget_python_id, f"Python {self.cfg.python_version}")
        console.print(
            "[yellow]Python installed/updated. IMPORTANT:[/yellow] reopen terminal and run setup again "
            "so the new python is on PATH."
        )
        # We stop here because PATH may not be updated in this process.
        raise SystemExit(0)

    def _ensure_venv(self) -> Path:
        root = self.cfg.project_root
        venv_path = root / self.cfg.venv_dir
        if not venv_path.exists():
            console.print(f"[cyan]Creating venv:[/cyan] {venv_path}")
            self._run([sys.executable, "-m", "venv", str(venv_path)])
        else:
            console.print(f"[green]Venv exists:[/green] {venv_path}")

        venv_python = venv_path / ("Scripts/python.exe")
        if not venv_python.exists():
            raise RuntimeError(f"Venv python not found: {venv_python}")
        return venv_python

    def _pip_install_requirements(self, venv_python: Path) -> None:
        root = self.cfg.project_root
        req = root / self.cfg.requirements_file
        if not req.exists():
            raise RuntimeError(f"requirements.txt not found: {req}")

        console.print("[cyan]Upgrading pip/setuptools/wheel...[/cyan]")
        self._run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])

        console.print("[cyan]Installing requirements...[/cyan]")
        self._run([str(venv_python), "-m", "pip", "install", "-r", str(req)])

    def _quick_check(self, venv_python: Path) -> None:
        console.print("[cyan]Quick check (torch/cuda, diffusers)...[/cyan]")
        self._run([str(venv_python), "-c", "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"])
        self._run([str(venv_python), "-c", "import diffusers; import transformers; print('diffusers OK', diffusers.__version__)"])