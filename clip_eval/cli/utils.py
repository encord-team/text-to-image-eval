from itertools import chain, product

import click
from InquirerPy import inquirer as inq
from InquirerPy.base.control import Choice

from clip_eval.common.data_models import EmbeddingDefinition
from clip_eval.dataset.provider import dataset_provider
from clip_eval.models.provider import model_provider
from clip_eval.utils import read_all_cached_embeddings


def _do_embedding_definition_selection(
    defs: list[EmbeddingDefinition], single: bool = False
) -> list[EmbeddingDefinition]:
    choices = [Choice(d, f"D: {d.dataset[:15]:18s} M: {d.model}") for d in defs]
    message = f"Please select the desired pair{'' if single else 's'}"
    definitions: list[EmbeddingDefinition] = inq.fuzzy(message, choices=choices, multiselect=True, vi_mode=True).execute()  # type: ignore
    return definitions


def _by_dataset(
    defs: list[EmbeddingDefinition] | dict[str, list[EmbeddingDefinition]]
) -> list[EmbeddingDefinition]:
    if isinstance(defs, list):
        defs_list = defs
        defs = {}
        for d in defs_list:
            defs.setdefault(d.dataset, []).append(d)

    choices = sorted(
        [
            Choice(v, f"D: {k[:15]:18s} M: {', '.join([d.model for d in v])}")
            for k, v in defs.items()
            if len(v)
        ],
        key=lambda c: len(c.value),
    )
    message = f"Please select dataset"
    definitions: list[EmbeddingDefinition] = inq.fuzzy(message, choices=choices, multiselect=False, vi_mode=True).execute()  # type: ignore
    return definitions


def select_existing_embedding_definitions(
    by_dataset: bool = False,
) -> list[EmbeddingDefinition]:
    defs = read_all_cached_embeddings(as_list=True)

    if by_dataset:
        # Subset definitions to specific dataset
        defs = _by_dataset(defs)

    return _do_embedding_definition_selection(defs)


def select_from_all_embedding_definitions(
    include_existing: bool = False, by_dataset: bool = False
) -> list[EmbeddingDefinition]:
    existing = set(read_all_cached_embeddings(as_list=True))

    models = model_provider.list_model_names()
    datasets = dataset_provider.list_dataset_names()

    defs = [
        EmbeddingDefinition(dataset=d, model=m) for d, m in product(datasets, models)
    ]
    if not include_existing:
        defs = [d for d in defs if d not in existing]

    if by_dataset:
        defs = _by_dataset(defs)

    return _do_embedding_definition_selection(defs)
