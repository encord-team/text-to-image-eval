from src.constants import PROJECT_PATHS
from src.types.data_models import EmbeddingDefinition

KEEP_CHARACTERS = set([".", "_", " "])
REPLACE_CHARACTERS = {" ": "_"}


def safe_str(unsafe: str) -> str:
    if not isinstance(unsafe, str):
        raise ValueError(f"{unsafe} ({type(unsafe)}) not a string")
    return "".join(REPLACE_CHARACTERS.get(c, c) for c in unsafe if c.isalnum() or c in KEEP_CHARACTERS).rstrip()


def read_all_cached_embeddings() -> dict[str, list[EmbeddingDefinition]]:
    """
    Reads existing embedding definitions from the cache directory.
    Returns: a dictionary of <dataset, [embeddings]> where the list is over models.
    """
    return {
        d.name: [
            EmbeddingDefinition(dataset=d.name, model=m.stem) for m in d.iterdir() if m.is_file() and m.suffix == ".npz"
        ]
        for d in PROJECT_PATHS.EMBEDDINGS.iterdir()
        if d.is_dir()
    }


if __name__ == "__main__":
    print(read_all_cached_embeddings())
