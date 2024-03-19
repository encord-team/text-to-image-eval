from typing import Annotated, Optional

import matplotlib.pyplot as plt
from typer import Option, Typer

from clip_eval.common.data_models import EmbeddingDefinition
from clip_eval.utils import read_all_cached_embeddings

from .utils import (
    select_existing_embedding_definitions,
    select_from_all_embedding_definitions,
)

cli = Typer(name="clip-eval", no_args_is_help=True, rich_markup_mode="markdown")


@cli.command(
    "build",
    help="""Build embeddings.
If no argumens are given, you will be prompted to select a combination of dataset and model(s).
You can use [TAB] to select multiple combinations and execute them sequentially.
 """,
)
def build_command(
    model_dataset: Annotated[str, Option(help="model, dataset pair delimited by model/dataset")] = "",
    include_existing: Annotated[
        bool,
        Option(help="Show also options for which the embeddings have been computed already"),
    ] = False,
    by_dataset: Annotated[
        bool,
        Option(help="Select dataset first, then model. Will only work if `model_dataset` not specified."),
    ] = False,
):
    if len(model_dataset) > 0:
        if model_dataset.count("/") != 1:
            raise ValueError("model dataset must contain only 1 /")
        model, dataset = model_dataset.split("/")
        definitions = [EmbeddingDefinition(model=model, dataset=dataset)]
    else:
        definitions = select_from_all_embedding_definitions(
            include_existing=include_existing,
            by_dataset=by_dataset,
        )

    for embd_defn in definitions:
        try:
            embeddings = embd_defn.build_embeddings()
            print("Made embedding successfully")
            embd_defn.save_embeddings(embeddings=embeddings)
            print("Saved embedding to file successfully at", embd_defn.embedding_path)
        except Exception as e:
            print(f"Failed to build embeddings for this bastard: {embd_defn}")
            print(e)
            import traceback

            traceback.print_exc()


@cli.command(
    "evaluate",
    help="""Evaluate embedding performance.
For this two work, you should have already run the `build` command for the model/dataset of interest.
""",
)
def evaluate_embeddings(
    model_datasets: Annotated[
        Optional[list[str]],
        Option(help="Specify specific combinations of models and datasets"),
    ] = None,
    is_all: Annotated[bool, Option(help="Evaluate all models.")] = False,
    save: Annotated[bool, Option(help="Save evaluation results to csv")] = False,
):
    from clip_eval.evaluation import (
        LinearProbeClassifier,
        WeightedKNNClassifier,
        ZeroShotClassifier,
    )
    from clip_eval.evaluation.evaluator import export_evaluation_to_csv, run_evaluation

    model_datasets = model_datasets or []

    if is_all:
        defns = read_all_cached_embeddings(as_list=True)
    elif len(model_datasets) > 0:
        # Error could be localised better
        if not all([model_dataset.count("/") == 1 for model_dataset in model_datasets]):
            raise ValueError("All model,dataset pairs must be presented as MODEL/DATASET")
        model_dataset_pairs = [model_dataset.split("/") for model_dataset in model_datasets]
        defns = [
            EmbeddingDefinition(model=model_dataset[0], dataset=model_dataset[1])
            for model_dataset in model_dataset_pairs
        ]
    else:
        defns = select_existing_embedding_definitions()

    models = [ZeroShotClassifier, LinearProbeClassifier, WeightedKNNClassifier]
    performances = run_evaluation(models, defns)
    if save:
        export_evaluation_to_csv(defns, performances)


@cli.command(
    "animate",
    help="""Animate 2D embeddings from two different models on the same dataset.
The interface will prompt you to choose which embeddings you want to use.
""",
)
def animate_embeddings(
    interactive: Annotated[bool, Option(help="Interactive plot instead of animation")] = False,
    reduction: Annotated[str, Option(help="Reduction type [pca, tsne, umap (default)")] = "umap",
):
    from clip_eval.plotting.animation import build_animation, save_animation_to_file

    defs = select_existing_embedding_definitions(by_dataset=True, count=2)
    res = build_animation(defs[0], defs[1], interactive=interactive, reduction=reduction)

    if res is None:
        plt.show()
    else:
        save_animation_to_file(res, *defs)


@cli.command("list", help="List models and datasets. By default, only cached pairs are listed.")
def list_models_datasets(
    all: Annotated[
        bool,
        Option(help="List all models and dataset that are available via the tool."),
    ] = False,
):
    from clip_eval.dataset.provider import dataset_provider
    from clip_eval.models import model_provider

    if all:
        datasets = dataset_provider.list_dataset_names()
        models = model_provider.list_model_names()
        print(f"Available datasets are: {', '.join(datasets)}")
        print(f"Available models are: {', '.join(models)}")
        return

    defns = read_all_cached_embeddings(as_list=True)
    print(f"Available model_dataset pairs: {', '.join([str(defn) for defn in defns])}")


if __name__ == "__main__":
    cli()
