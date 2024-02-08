import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CACHE_PATH = Path(os.environ.get("CLIP_CACHE_PATH", ".cache"))
