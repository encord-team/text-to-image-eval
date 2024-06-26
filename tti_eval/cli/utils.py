from itertools import product
from typing import Literal, overload

from InquirerPy import inquirer as inq
from InquirerPy.base.control import Choice
from natsort import natsorted, ns

from tti_eval.common import EmbeddingDefinition
from tti_eval.dataset import DatasetProvider
from tti_eval.model import ModelProvider
from tti_eval.utils import read_all_cached_embeddings


@overload
def _do_embedding_definition_selection(
    defs: list[EmbeddingDefinition], allow_multiple: Literal[True] = True
) -> list[EmbeddingDefinition]:
    ...


@overload
def _do_embedding_definition_selection(
    defs: list[EmbeddingDefinition], allow_multiple: Literal[False]
) -> EmbeddingDefinition:
    ...


def _do_embedding_definition_selection(
    defs: list[EmbeddingDefinition],
    allow_multiple: bool = True,
) -> list[EmbeddingDefinition] | EmbeddingDefinition:
    sorted_defs = natsorted(defs, key=lambda x: (x.dataset, x.model), alg=ns.IGNORECASE)
    choices = [Choice(d, f"D: {d.dataset[:15]:18s} M: {d.model}") for d in sorted_defs]
    message = "Please select the desired pairs" if allow_multiple else "Please select a pair"
    definitions = inq.fuzzy(
        message,
        choices=choices,
        multiselect=allow_multiple,
        vi_mode=True,
    ).execute()  # type: ignore
    return definitions


def _by_dataset(defs: list[EmbeddingDefinition] | dict[str, list[EmbeddingDefinition]]) -> list[EmbeddingDefinition]:
    if isinstance(defs, list):
        defs_list = defs
        defs = {}
        for d in defs_list:
            defs.setdefault(d.dataset, []).append(d)

    choices = sorted(
        [Choice(v, f"D: {k[:15]:18s} M: {', '.join([d.model for d in v])}") for k, v in defs.items() if len(v)],
        key=lambda c: len(c.value),
    )
    message = "Please select a dataset"
    definitions: list[EmbeddingDefinition] = inq.fuzzy(
        message, choices=choices, multiselect=False, vi_mode=True
    ).execute()  # type: ignore
    return definitions


def parse_raw_embedding_definitions(raw_embedding_definitions: list[str]) -> list[EmbeddingDefinition]:
    if not all([model_dataset.count("/") == 1 for model_dataset in raw_embedding_definitions]):
        raise ValueError("All (model, dataset) pairs must be presented as MODEL/DATASET")
    model_dataset_pairs = [model_dataset.split("/") for model_dataset in raw_embedding_definitions]
    return [
        EmbeddingDefinition(model=model_dataset[0], dataset=model_dataset[1]) for model_dataset in model_dataset_pairs
    ]


def select_existing_embedding_definitions(
    by_dataset: bool = False,
    count: int | None = None,
) -> list[EmbeddingDefinition]:
    defs = read_all_cached_embeddings(as_list=True)

    if by_dataset:
        # Subset definitions to specific dataset
        defs = _by_dataset(defs)

    if count is None:
        return _do_embedding_definition_selection(defs)
    else:
        return [_do_embedding_definition_selection(defs, allow_multiple=False) for _ in range(count)]


def select_from_all_embedding_definitions(
    include_existing: bool = False, by_dataset: bool = False
) -> list[EmbeddingDefinition]:
    existing = set(read_all_cached_embeddings(as_list=True))

    models = ModelProvider.list_model_titles()
    datasets = DatasetProvider.list_dataset_titles()

    defs = [EmbeddingDefinition(dataset=d, model=m) for d, m in product(datasets, models)]
    if not include_existing:
        defs = [d for d in defs if d not in existing]

    if by_dataset:
        defs = _by_dataset(defs)

    return _do_embedding_definition_selection(defs)
