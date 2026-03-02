# services/update_service.py
import io
import zipfile
import requests
from pathlib import Path


class UpdateService:
    def __init__(self, owner: str, repo: str, branch: str = "main"):
        self.owner = owner
        self.repo = repo
        self.branch = branch

    def update_from_github_zip(self, project_root: str) -> str:
        root = Path(project_root).resolve()
        url = f"https://github.com/{self.owner}/{self.repo}/archive/refs/heads/{self.branch}.zip"

        r = requests.get(url, timeout=60)
        r.raise_for_status()

        z = zipfile.ZipFile(io.BytesIO(r.content))
        top_folder = z.namelist()[0].split("/")[0]  # repo-branch

        skip = {"outputs", ".venv", "__pycache__", ".git"}

        for name in z.namelist():
            if name.endswith("/"):
                continue
            rel = Path(name).relative_to(top_folder)
            if rel.parts and rel.parts[0] in skip:
                continue

            out_path = root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(z.read(name))

        return "OK"