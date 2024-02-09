import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_CACHE_PATH = Path(os.environ.get("CLIP_CACHE_PATH", ".cache"))


class PROJECT_PATHS:
    EMBEDDINGS = _CACHE_PATH / "embeddings"
    REDUCTIONS = _CACHE_PATH / "reductions"


class NPZ_KEYS:
    IMAGE_EMBEDDINGS = "image_embeddings"
    CLASS_EMBEDDINGS = "class_embeddings"
    LABELS = "labels"
