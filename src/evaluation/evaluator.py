import csv
from datetime import datetime

import numpy as np

from src.common.data_models import EmbeddingDefinition, Embeddings
from src.constants import OUTPUT_PATH
from src.evaluation import (
    ClassificationModel,
    LinearProbeClassifier,
    WeightedKNNClassifier,
    ZeroShotClassifier,
)
from src.utils import read_all_cached_embeddings


def run_evaluation(
    classifiers: list[type[ClassificationModel]],
    embedding_definitions: list[EmbeddingDefinition],
    seed: int = 42,
    train_split: float = 0.7,  # TODO: This is very much out of the blue
) -> list[dict[str, float]]:
    embeddings_performance: list[dict[str, float]] = []
    for def_ in embedding_definitions:
        embeddings = def_.load_embeddings()

        if embeddings is None:
            print(f"No embedding found for {def_}")
            embeddings_performance.append({})
            continue

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
            if classifier_type == ZeroShotClassifier and embeddings.classes is None:
                continue
            classifier = classifier_type(**model_args)
            probs, y_hat = classifier.predict(Embeddings(images=validation_embeddings, labels=validation_labels))
            acc = (y_hat == validation_labels).astype(float).mean()
            print(def_, classifier.title, acc)
            classifier_performance[classifier.title] = acc
        embeddings_performance.append(classifier_performance)
    return embeddings_performance


def export_evaluation_to_csv(
    embedding_definitions: list[EmbeddingDefinition],
    embeddings_performance: list[dict[str, float]],
) -> None:
    ts = datetime.now()
    results_file = OUTPUT_PATH.EVALUATIONS / f"eval_{ts.isoformat()}.csv"
    results_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure that parent folder exists

    headers = ["Model", "Dataset", "Classifier", "Accuracy"]
    with open(results_file.as_posix(), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for def_, perf in zip(embedding_definitions, embeddings_performance, strict=True):
            def_: EmbeddingDefinition
            for classifier_title, accuracy in perf.items():
                writer.writerow([def_.model, def_.dataset, classifier_title, accuracy])


if __name__ == "__main__":
    models = [ZeroShotClassifier, LinearProbeClassifier, WeightedKNNClassifier]
    defs = [d for k, v in read_all_cached_embeddings().items() for d in v]
    print(defs)
    performances = run_evaluation(models, defs)
    export_evaluation_to_csv(defs, performances)
    print(performances)
