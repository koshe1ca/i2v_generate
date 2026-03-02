from __future__ import annotations

from pathlib import Path
import typer
from rich.console import Console

from services.install_service import InstallService, SetupConfig

app = typer.Typer()
console = Console()


@app.command()
def setup():
    """
    One-click setup for Windows:
    installs python (via winget), creates venv, installs deps.
    """
    root = Path(__file__).resolve().parents[1]
    svc = InstallService(SetupConfig(project_root=root))
    svc.setup_all()


if __name__ == "__main__":
    app()