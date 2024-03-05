import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# If the cache directory is not explicitly specified, use the `.cache` directory located in the project's root.
CACHE_PATH = Path(os.environ.get("CLIP_CACHE_PATH", Path(__file__).parent.parent / ".cache"))
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
