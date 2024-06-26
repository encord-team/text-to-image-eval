import csv
from datetime import datetime

from natsort import natsorted, ns
from tabulate import tabulate
from tqdm.auto import tqdm

from tti_eval.common import EmbeddingDefinition, Split
from tti_eval.constants import OUTPUT_PATH
from tti_eval.evaluation import (
    EvaluationModel,
    I2IRetrievalEvaluator,
    LinearProbeClassifier,
    WeightedKNNClassifier,
    ZeroShotClassifier,
)
from tti_eval.utils import read_all_cached_embeddings


def print_evaluation_results(
    results: dict[EmbeddingDefinition, dict[str, float]],
    evaluation_model_title: str,
):
    defs = list(results.keys())
    model_names = natsorted(set(map(lambda d: d.model, defs)), alg=ns.IGNORECASE)
    dataset_names = natsorted(set(map(lambda d: d.dataset, defs)), alg=ns.IGNORECASE)

    table: list[list[float | str]] = [
        ["Model/Dataset"] + dataset_names,
    ] + [[m] + ["-"] * len(dataset_names) for m in model_names]

    def set_score(d: EmbeddingDefinition, s: float):
        row = model_names.index(d.model) + 1
        col = dataset_names.index(d.dataset) + 1
        table[row][col] = f"{s:.4f}"

    for d, res in results.items():
        s = res.get(evaluation_model_title)
        if s:
            set_score(d, res[evaluation_model_title])

    print(f"{'='*5} {evaluation_model_title} {'=' * 5}")
    print(tabulate(table))


def run_evaluation(
    evaluators: list[type[EvaluationModel]],
    embedding_definitions: list[EmbeddingDefinition],
) -> dict[EmbeddingDefinition, dict[str, float]]:
    embeddings_performance: dict[EmbeddingDefinition, dict[str, float]] = {}
    used_evaluators: set[str] = set()

    for def_ in tqdm(embedding_definitions, desc="Evaluating models", leave=False):
        train_embeddings = def_.load_embeddings(Split.TRAIN)
        validation_embeddings = def_.load_embeddings(Split.VALIDATION)

        if train_embeddings is None:
            print(f"No train embeddings were found for {def_}")
            continue
        if validation_embeddings is None:
            print(f"No validation embeddings were found for {def_}")
            continue

        evaluator_performance: dict[str, float] = embeddings_performance.setdefault(def_, {})
        for evaluator_type in evaluators:
            if evaluator_type == ZeroShotClassifier and train_embeddings.classes is None:
                continue
            evaluator = evaluator_type(
                train_embeddings=train_embeddings,
                validation_embeddings=validation_embeddings,
            )
            evaluator_performance[evaluator.title()] = evaluator.evaluate()
            used_evaluators.add(evaluator.title())

    for evaluator_type in evaluators:
        evaluator_title = evaluator_type.title()
        if evaluator_title in used_evaluators:
            print_evaluation_results(embeddings_performance, evaluator_title)
    return embeddings_performance


def export_evaluation_to_csv(embeddings_performance: dict[EmbeddingDefinition, dict[str, float]]) -> None:
    ts = datetime.now()
    results_file = OUTPUT_PATH.EVALUATIONS / f"eval_{ts.isoformat()}.csv"
    results_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure that parent folder exists

    headers = ["Model", "Dataset", "Classifier", "Accuracy"]
    with open(results_file.as_posix(), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

        for def_, perf in embeddings_performance.items():
            def_: EmbeddingDefinition
            for classifier_title, accuracy in perf.items():
                writer.writerow([def_.model, def_.dataset, classifier_title, accuracy])
    print(f"Evaluation results exported to `{results_file.as_posix()}`")


if __name__ == "__main__":
    models = [ZeroShotClassifier, LinearProbeClassifier, WeightedKNNClassifier, I2IRetrievalEvaluator]
    defs = read_all_cached_embeddings(as_list=True)
    print(defs)
    performances = run_evaluation(models, defs)
    export_evaluation_to_csv(performances)
    print(performances)
