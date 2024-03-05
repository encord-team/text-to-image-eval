import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CACHE_PATH = Path(os.environ.get("CLIP_CACHE_PATH", ".cache"))
_OUTPUT_PATH = Path(os.environ.get("OUTPUT_PATH", "output"))


class PROJECT_PATHS:
    EMBEDDINGS = CACHE_PATH / "embeddings"
    REDUCTIONS = CACHE_PATH / "reductions"


class NPZ_KEYS:
    IMAGE_EMBEDDINGS = "image_embeddings"
    CLASS_EMBEDDINGS = "class_embeddings"
    LABELS = "labels"


class OUTPUT_PATH:
    ANIMATIONS = _OUTPUT_PATH / "animations"
    EVALUATIONS = _OUTPUT_PATH / "evaluations"
