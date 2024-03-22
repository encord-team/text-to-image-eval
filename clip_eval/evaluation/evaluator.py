import csv
from datetime import datetime
from typing import Literal

import numpy as np
from natsort import natsorted
from tabulate import tabulate

from clip_eval.common.data_models import EmbeddingDefinition, Embeddings
from clip_eval.constants import OUTPUT_PATH
from clip_eval.evaluation import (
    ClassificationModel,
    LinearProbeClassifier,
    WeightedKNNClassifier,
    ZeroShotClassifier,
)
from clip_eval.utils import read_all_cached_embeddings


def print_evaluation_results(
    results: dict[EmbeddingDefinition, dict[str, float]],
    classifier_column: (Literal["linear_probe"] | Literal["zero_shot"] | Literal["wKNN"]) = "linear_probe",
):
    defs = list(results.keys())
    model_names = natsorted(set(map(lambda d: d.model, defs)))
    dataset_names = natsorted(set(map(lambda d: d.dataset, defs)))

    table: list[list[float | str]] = [
        ["Model/Dataset"] + dataset_names,
    ] + [[m] + ["-"] * len(dataset_names) for m in model_names]

    def set_score(d: EmbeddingDefinition, s: float):
        row = model_names.index(d.model) + 1
        col = dataset_names.index(d.dataset) + 1
        table[row][col] = f"{s:.4f}"

    for d, res in results.items():
        s = res.get(classifier_column)
        if s:
            set_score(d, res[classifier_column])

    print(f"{'='*5} {classifier_column} {'='*5}")
    print(tabulate(table))


def run_evaluation(
    classifiers: list[type[ClassificationModel]],
    embedding_definitions: list[EmbeddingDefinition],
    seed: int = 42,
    train_split: float = 0.7,  # TODO: This is very much out of the blue
) -> dict[EmbeddingDefinition, dict[str, float]]:
    embeddings_performance: dict[EmbeddingDefinition, dict[str, float]] = {}
    model_keys: set[str] = set()

    for def_ in embedding_definitions:
        embeddings = def_.load_embeddings()

        if embeddings is None:
            print(f"No embedding found for {def_}")
            continue

        n, _ = embeddings.images.shape

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
        classifier_performance: dict[str, float] = embeddings_performance.setdefault(def_, {})
        for classifier_type in classifiers:
            if classifier_type == ZeroShotClassifier and embeddings.classes is None:
                continue
            classifier = classifier_type(**model_args)
            acc = classifier.evaluate(Embeddings(images=validation_embeddings, labels=validation_labels))
            classifier_performance[classifier.title] = acc
            model_keys.add(classifier.title)

    for n in model_keys:
        print_evaluation_results(embeddings_performance, n)

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
    defs = read_all_cached_embeddings(as_list=True)
    print(defs)
    performances = run_evaluation(models, defs)
    export_evaluation_to_csv(defs, performances)
    print(performances)
