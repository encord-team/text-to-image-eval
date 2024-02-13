import numpy as np

from src.common.data_models import EmbeddingDefinition, Embeddings
from src.evaluation import (
    ClassificationModel,
    LinearProbeClassifier,
    WeightedKNNClassifier,
    ZeroShotClassifier,
)
from src.utils import read_all_cached_embeddings


def run_evaluation(
    classifiers: ClassificationModel,
    embedding_definitions: list[EmbeddingDefinition],
    seed: int = 42,
    train_split: float = 0.7,  # TODO: This is very much out of the blue
) -> list[dict[str, float]]:
    embeddings_performance: list[dict[str, float]] = []
    for def_ in embedding_definitions:
        embeddings = def_.load_embeddings()

        if embeddings is None:
            # FIXME: Store empty result
            return None

        n, d = embeddings.images.shape

        np.random.seed(seed)
        selection = np.random.permutation(n)
        train_size = int(n * train_split)

        train_embeddings = embeddings.images[selection[:train_size]]
        train_labels = embeddings.labels[selection[:train_size]]

        validation_embeddings = embeddings.images[selection[train_size:]]
        validation_labels = embeddings.labels[selection[train_size:]]

        model_args = {
            "embeddings": train_embeddings,
            "labels": train_labels,
            "class_embeddings": embeddings.classes,
        }
        classifier_performance = {}
        for classifier_type in classifiers:
            classifier = classifier_type(**model_args)
            probs, y_hat = classifier.predict(Embeddings(images=validation_embeddings, labels=validation_labels))
            acc = (y_hat == validation_labels).astype(float).mean()
            print(def_, classifier.title, acc)
            classifier_performance[classifier.title] = acc
            # FIXME: Store the results
        print(classifier_performance)
        embeddings_performance.append(classifier_performance)
    # FIXME: Report the results
    return embeddings_performance


if __name__ == "__main__":
    models = [ZeroShotClassifier, LinearProbeClassifier, WeightedKNNClassifier]
    defs = [d for k, v in read_all_cached_embeddings().items() for d in v]
    print(len(defs))
    run_evaluation(models, defs)
