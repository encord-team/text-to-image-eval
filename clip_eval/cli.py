from typing import Annotated

import matplotlib.pyplot as plt
from typer import Argument, Option, Typer

from clip_eval.common.data_models import EmbeddingDefinition
from clip_eval.utils import read_all_cached_embeddings

cli = Typer(name="clip-eval", no_args_is_help=True, rich_markup_mode="markdown")


@cli.command("build", help="Build embeddings")
def build_command(
    model_dataset: Annotated[
        str, Argument(help="model, dataset pair delimited by model/dataset")
    ]
):
    if model_dataset.count("/") != 1:
        raise ValueError("model dataset must contain only 1 /")
    model, dataset = model_dataset.split("/")
    embd_defn = EmbeddingDefinition(model=model, dataset=dataset)
    embeddings = embd_defn.build_embeddings()
    print("Made embedding successfully")
    embd_defn.save_embeddings(embeddings=embeddings)
    print("Saved embedding to file successfully at", embd_defn.embedding_path)


@cli.command("evaluate", help="Evaluate embedding performance")
def evaluate_embeddings(
    model_datasets: Annotated[
        list[str] | None,
        Option(help="Specify specific combinations of models and datasets"),
    ] = None,
    is_all: Annotated[bool, Option(help="Evaluate all models.")] = False,
    save: Annotated[bool, Option(help="")] = False,
):
    from clip_eval.evaluation import (
        LinearProbeClassifier,
        WeightedKNNClassifier,
        ZeroShotClassifier,
    )
    from clip_eval.evaluation.evaluator import export_evaluation_to_csv, run_evaluation

    model_datasets = model_datasets or []
    if is_all:
        defns: list[EmbeddingDefinition] = [
            d for k, v in read_all_cached_embeddings().items() for d in v
        ]
    else:
        # Error could be localised better
        if not all([model_dataset.count("/") == 1 for model_dataset in model_datasets]):
            raise ValueError(
                "All model,dataset pairs must be presented as MODEL/DATASET"
            )
        model_dataset_pairs = [
            model_dataset.split("/") for model_dataset in model_datasets
        ]
        defns = [
            EmbeddingDefinition(model=model_dataset[0], dataset=model_dataset[1])
            for model_dataset in model_dataset_pairs
        ]
    models = [ZeroShotClassifier, LinearProbeClassifier, WeightedKNNClassifier]
    performances = run_evaluation(models, defns)
    if save:
        export_evaluation_to_csv(defns, performances)
    print(performances)


@cli.command(
    "animate", help="2 (shortened) model dataset pairs written as model/dataset"
)
def animate_embeddings(
    from_model_dataset: Annotated[str, Argument(help="model/dataset pair")],
    to_model_dataset: Annotated[str, Argument(help="model/dataset pair")],
):
    from clip_eval.plotting.animation import build_animation, save_animation_to_file

    # Error could be localised better
    model_datasets = [from_model_dataset, to_model_dataset]
    assert len(model_datasets) == 2
    if not all([model_dataset.count("/") == 1 for model_dataset in model_datasets]):
        raise ValueError("All model,dataset pairs must be presented as MODEL/DATASET")
    model_dataset_pairs = [model_dataset.split("/") for model_dataset in model_datasets]
    assert len(model_dataset_pairs) == 2 and len(model_dataset_pairs[0]) == 2
    defns = [
        EmbeddingDefinition(model=model_dataset[0], dataset=model_dataset[1])
        for model_dataset in model_dataset_pairs
    ]
    anim = build_animation(*defns)
    save_animation_to_file(anim, *defns)
    plt.show()


@cli.command("list", help="List models and datasets")
def list_models_datasets(
    all: Annotated[bool, Option(help="List all available models and dataset")] = False
):
    from clip_eval.dataset.provider import dataset_provider
    from clip_eval.models import model_provider

    if all:
        datasets = dataset_provider.list_dataset_names()
        models = model_provider.list_model_names()
        print(f"Available datasets are: {', '.join(datasets)}")
        print(f"Available models are: {', '.join(models)}")
        return

    defns: list[EmbeddingDefinition] = [
        d for k, v in read_all_cached_embeddings().items() for d in v
    ]
    print(f"Available model_dataset pairs: {', '.join([str(defn) for defn in defns])}")


if __name__ == "__main__":
    cli()
