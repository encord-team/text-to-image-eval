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


def select_existing_embedding_definitions(
    by_dataset: bool = False,
) -> list[EmbeddingDefinition]:
    edefs_by_dataset = read_all_cached_embeddings()

    if by_dataset or click.confirm("Choose by dataset?"):
        # Subset definitions to specific dataset
        choices = [
            Choice(v, f"D: {k[:15]:18s} M: {', '.join([d.model for d in v])}")
            for k, v in edefs_by_dataset.items()
            if len(v)
        ]
        message = f"Please select dataset"
        definitions: list[EmbeddingDefinition] = inq.fuzzy(message, choices=choices, multiselect=False, vi_mode=True).execute()  # type: ignore
    else:
        definitions = list(chain(*edefs_by_dataset.values()))

    return _do_embedding_definition_selection(definitions)


def select_from_all_embedding_definitions(
    include_existing: bool = False,
) -> list[EmbeddingDefinition]:
    existing = set(chain(*read_all_cached_embeddings().values()))

    models = model_provider.list_model_names()
    datasets = dataset_provider.list_dataset_names()

    all_defs = [
        EmbeddingDefinition(dataset=d, model=m) for d, m in product(datasets, models)
    ]
    if not include_existing:
        print("excluding")
        all_defs = list(filter(lambda x: x not in existing, all_defs))

    return _do_embedding_definition_selection(all_defs)
