# services/face_restore_service.py
from typing import List
from PIL import Image


class FaceRestoreService:
    """
    Заготовка.
    Позже подключим реальную реализацию CodeFormer/GPEN.
    Сейчас просто возвращает кадры как есть, чтобы архитектура была готова.
    """

    def restore(self, frames: List[Image.Image], method: str = "codeformer", strength: float = 0.6) -> List[Image.Image]:
        # TODO: подключим реально CodeFormer/GPEN
        return frames