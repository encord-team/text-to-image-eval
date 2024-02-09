import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CACHE_PATH = Path(os.environ.get("CLIP_CACHE_PATH", ".cache"))

NPZ_IMAGE_EMBEDDING_KEY = "image_embeddings"
NPZ_CLASS_EMBEDDING_KEY = "label_embeddings"
