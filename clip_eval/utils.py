import importlib.util
from itertools import chain
from typing import Literal, overload

from clip_eval.common.data_models import EmbeddingDefinition
from clip_eval.constants import PROJECT_PATHS


def load_class_from_path(module_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location(module_path, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


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
