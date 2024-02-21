import argparse

import matplotlib.pyplot as plt

from src.common.data_models import EmbeddingDefinition
from src.dataset.provider import dataset_provider
from src.evaluation import (
    LinearProbeClassifier,
    WeightedKNNClassifier,
    ZeroShotClassifier,
)
from src.evaluation.evaluator import export_evaluation_to_csv, run_evaluation
from src.models.CLIP_model import CLIPModel
from src.plotting.animation import build_animation, save_animation_to_file
from src.utils import read_all_cached_embeddings


def build_command(args):
    build_embedding(args.model_dataset)


def build_embedding(model_dataset: str):
    if model_dataset.count("/") != 1:
        raise ValueError("model dataset must contain only 1 /")
    model, dataset = model_dataset.split("/")
    embd_defn = EmbeddingDefinition(model=model, dataset=dataset)
    embeddings = embd_defn.build_embeddings()
    print("Made embedding successfully")
    embd_defn.save_embeddings(embeddings=embeddings)
    print("Saved embedding to file successfully at", embd_defn.embedding_path)


def evaluate_command(args):
    evaluate_embeddings(args.model_datasets, args.all, args.save)


def evaluate_embeddings(model_datasets: list[str] | None = None, is_all: bool = False, save: bool = False):
    if is_all:
        defns: list[EmbeddingDefinition] = [d for k, v in read_all_cached_embeddings().items() for d in v]
    else:
        # Error could be localised better
        if not all([model_dataset.count("/") == 1 for model_dataset in model_datasets]):
            raise ValueError("All model,dataset pairs must be presented as MODEL/DATASET")
        model_dataset_pairs = [model_dataset.split("/") for model_dataset in model_datasets]
        defns = [
            EmbeddingDefinition(model=model_dataset[0], dataset=model_dataset[1])
            for model_dataset in model_dataset_pairs
        ]
    models = [ZeroShotClassifier, LinearProbeClassifier, WeightedKNNClassifier]
    performances = run_evaluation(models, defns)
    if save:
        export_evaluation_to_csv(defns, performances)
    print(performances)


def animate_command(args):
    animate_embeddings(args.model_datasets)


def animate_embeddings(model_datasets: list[str]):
    # Error could be localised better
    assert len(model_datasets) == 2
    if not all([model_dataset.count("/") == 1 for model_dataset in model_datasets]):
        raise ValueError("All model,dataset pairs must be presented as MODEL/DATASET")
    model_dataset_pairs = [model_dataset.split("/") for model_dataset in model_datasets]
    assert len(model_dataset_pairs) == 2 and len(model_dataset_pairs[0]) == 2
    defns = [
        EmbeddingDefinition(model=model_dataset[0], dataset=model_dataset[1]) for model_dataset in model_dataset_pairs
    ]
    anim = build_animation(*defns)
    save_animation_to_file(anim, *defns)
    plt.show()


def list_command(args):
    list_models_datasets(args.all)


def list_models_datasets(all: bool = False):
    if all:
        datasets = dataset_provider.list_dataset_names()
        models = CLIPModel.list_models()
        print(f"Available datasets are: {', '.join(datasets)}")
        print(f"Available models are: {', '.join(models)}")
        return
    else:
        defns: list[EmbeddingDefinition] = [d for k, v in read_all_cached_embeddings().items() for d in v]
        print(f"Available model_dataset pairs: {', '.join([str(defn) for defn in defns])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multiclip builder and Benchmarker")
    subparsers = parser.add_subparsers()

    parser_build = subparsers.add_parser("build", help="build embeddings")

    parser_build.add_argument(
        "model_dataset", type=str, help="model, dataset pair delimited by model/dataset", default=None
    )
    parser_build.set_defaults(func=build_command)

    parser_evaluate = subparsers.add_parser("evaluate", help="evaluate embeddings")
    parser_evaluate.add_argument("--all", action="store_true", help="Evaluate all models")
    parser_evaluate.add_argument(
        "model_datasets",
        metavar="model/dataset",
        nargs="+",
        help="(shortened) model dataset pairs written as model/dataset",
    )
    parser_evaluate.add_argument("--save", action="store_true", help="Save report to CSV")
    parser_evaluate.set_defaults(func=evaluate_command)

    parser_animation = subparsers.add_parser("animate", help="animate between two embeddings")
    parser_animation.add_argument(
        "model_datasets",
        metavar="model/dataset",
        nargs=2,
        help="2 (shortened) model dataset pairs written as model/dataset",
    )

    parser_animation.set_defaults(func=animate_command)

    parser_list = subparsers.add_parser("list", help="List available dataset, model pairs")
    parser_list.add_argument("--all", action="store_true", help="List all available models and dataset")
    parser_list.set_defaults(func=list_command)
    args = parser.parse_args()
    args.func(args)
