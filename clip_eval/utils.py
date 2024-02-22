from clip_eval.common.data_models import EmbeddingDefinition
from clip_eval.constants import PROJECT_PATHS


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
