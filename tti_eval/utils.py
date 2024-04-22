from itertools import chain
from typing import Literal, overload

from clip_eval.common import EmbeddingDefinition
from clip_eval.constants import PROJECT_PATHS


@overload
def read_all_cached_embeddings(as_list: Literal[True]) -> list[EmbeddingDefinition]:
    ...


@overload
def read_all_cached_embeddings(as_list: Literal[False] = False) -> dict[str, list[EmbeddingDefinition]]:
    ...


def read_all_cached_embeddings(
    as_list: bool = False,
) -> dict[str, list[EmbeddingDefinition]] | list[EmbeddingDefinition]:
    """
    Reads existing embedding definitions from the cache directory.
    Returns: a dictionary of <dataset, [embeddings]> where the list is over models.
    """
    if not PROJECT_PATHS.EMBEDDINGS.exists():
        return dict()

    defs_dict = {
        d.name: list(
            {
                EmbeddingDefinition(dataset=d.name, model=m.stem.rsplit("_", maxsplit=1)[0])
                for m in d.iterdir()
                if m.is_file() and m.suffix == ".npz"
            }
        )
        for d in PROJECT_PATHS.EMBEDDINGS.iterdir()
        if d.is_dir()
    }
    if as_list:
        return list(chain(*defs_dict.values()))
    return defs_dict


if __name__ == "__main__":
    print(read_all_cached_embeddings())
