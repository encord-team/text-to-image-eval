from types.data_models import EmbeddingDefinition

KEEP_CHARACTERS = set([".", "_", " "])
REPLACE_CHARACTERS = {" ": "_"}


def safe_str(unsafe: str) -> str:
    if not isinstance(unsafe, str):
        raise ValueError(f"{unsafe} ({type(unsafe)}) not a string")
    return "".join(REPLACE_CHARACTERS.get(c, c) for c in unsafe if c.isalnum() or c in KEEP_CHARACTERS).rstrip()


def read_cached_embeddings() -> dict[str, list[EmbeddingDefinition]]:
    """
    Reads existing embedding definitions from the cache directory.
    Returns: a dictionary of <dataset, [embeddings]> where the list is over models.

    """
    pass
