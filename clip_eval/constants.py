import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# If the cache directory is not explicitly specified, use the `.cache` directory located in the project's root.
_CLIP_EVAL_ROOT_DIR = Path(__file__).parent.parent
CACHE_PATH = Path(os.environ.get("CLIP_EVAL_CACHE_PATH", _CLIP_EVAL_ROOT_DIR / ".cache"))
_OUTPUT_PATH = Path(os.environ.get("CLIP_EVAL_OUTPUT_PATH", _CLIP_EVAL_ROOT_DIR / "output"))
_SOURCES_PATH = _CLIP_EVAL_ROOT_DIR / "sources"


class PROJECT_PATHS:
    EMBEDDINGS = CACHE_PATH / "embeddings"
    MODELS = CACHE_PATH / "models"
    REDUCTIONS = CACHE_PATH / "reductions"


class NPZ_KEYS:
    IMAGE_EMBEDDINGS = "image_embeddings"
    CLASS_EMBEDDINGS = "class_embeddings"
    LABELS = "labels"


class OUTPUT_PATH:
    ANIMATIONS = _OUTPUT_PATH / "animations"
    EVALUATIONS = _OUTPUT_PATH / "evaluations"


class SOURCES_PATH:
    DATASET_TYPES = _SOURCES_PATH / "datasets" / "types"
    DATASET_INSTANCE_DEFINITIONS = _SOURCES_PATH / "datasets" / "instances"
    MODEL_TYPES = _SOURCES_PATH / "models" / "types"
    MODEL_INSTANCE_DEFINITIONS = _OUTPUT_PATH / "models" / "instances"
