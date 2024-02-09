from typing import Type

import numpy as np

from src.types.data_models import EmbeddingDefinition

from .base import ClassificationModel


def run_evaluation(
    models: list[Type[ClassificationModel]],
    embedding_definitions: list[EmbeddingDefinition],
    seed: int = 42,
    train_split: float = 0.7,  # TODO: This is very much out of the blue
):
    for def_ in embedding_definitions:
        embeddings = def_.load_embeddings()
        if embeddings is None:
            # TODO potentially store empty result
            return None

        n, d = embeddings.images.shape

        np.random.seed(seed)
        selection = np.random.permutation(n)
        train_size = int(n * train_split)

        train_embeddings = embeddings.images[selection[:train_size]]
        train_labels = embeddings.labels[selection[:train_size]]
        validation_embeddings = embeddings.images[selection[train_size:]]
        validation_embeddings = embeddings.labels[selection[train_size:]]

        model_args = {}
        for model in models:
            # TODO how do we instantiate the models?
            model(**model_args)


if __name__ == "__main__":
    run_evaluation()
