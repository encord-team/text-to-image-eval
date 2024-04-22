from typing import Annotated, Optional

import matplotlib.pyplot as plt
from typer import Option, Typer

from clip_eval.common import Split
from clip_eval.compute import compute_embeddings_from_definition
from clip_eval.utils import read_all_cached_embeddings

from .utils import (
    parse_raw_embedding_definitions,
    select_existing_embedding_definitions,
    select_from_all_embedding_definitions,
)

cli = Typer(name="clip-eval", no_args_is_help=True, rich_markup_mode="markdown")


@cli.command(
    "build",
    help="""Build embeddings.
If no arguments are given, you will be prompted to select the model and dataset combinations for generating embeddings.
You can use [TAB] to select multiple combinations and execute them sequentially.
 """,
)
def build_command(
    model_datasets: Annotated[
        Optional[list[str]],
        Option(
            "--model-dataset",
            help="Specify a model and dataset combination. Can be used multiple times. "
            "(model, dataset) pairs must be presented as 'MODEL/DATASET'.",
        ),
    ] = None,
    include_existing: Annotated[
        bool,
        Option(help="Show combinations for which the embeddings have already been computed."),
    ] = False,
    by_dataset: Annotated[
        bool,
        Option(help="Select dataset first, then model. Will only work if `model_dataset` is not specified."),
    ] = False,
):
    if len(model_datasets) > 0:
        definitions = parse_raw_embedding_definitions(model_datasets)
    else:
        definitions = select_from_all_embedding_definitions(
            include_existing=include_existing,
            by_dataset=by_dataset,
        )

    splits = [Split.TRAIN, Split.VALIDATION]
    for embd_defn in definitions:
        for split in splits:
            try:
                embeddings = compute_embeddings_from_definition(embd_defn, split)
                embd_defn.save_embeddings(embeddings=embeddings, split=split, overwrite=True)
                print(f"Embeddings saved successfully to file at `{embd_defn.embedding_path(split)}`")
            except Exception as e:
                print(f"Failed to build embeddings for this bastard: {embd_defn}")
                print(e)
                import traceback

                traceback.print_exc()


@cli.command(
    "evaluate",
    help="""Evaluate embeddings performance.
If no arguments are given, you will be prompted to select the model and dataset combinations to evaluate.
Only (model, dataset) pairs whose embeddings have been built will be available for evaluation.
You can use [TAB] to select multiple combinations and execute them sequentially.
""",
)
def evaluate_embeddings(
    model_datasets: Annotated[
        Optional[list[str]],
        Option(
            "--model-dataset",
            help="Specify a model and dataset combination. Can be used multiple times. "
            "(model, dataset) pairs must be presented as 'MODEL/DATASET'.",
        ),
    ] = None,
    all_: Annotated[bool, Option("--all", "-a", help="Evaluate all models.")] = False,
    save: Annotated[bool, Option("--save", "-s", help="Save evaluation results to a CSV file.")] = False,
):
    from clip_eval.evaluation import (
        I2IRetrievalEvaluator,
        LinearProbeClassifier,
        WeightedKNNClassifier,
        ZeroShotClassifier,
    )
    from clip_eval.evaluation.evaluator import export_evaluation_to_csv, run_evaluation

    model_datasets = model_datasets or []

    if all_:
        definitions = read_all_cached_embeddings(as_list=True)
    elif len(model_datasets) > 0:
        definitions = parse_raw_embedding_definitions(model_datasets)
    else:
        definitions = select_existing_embedding_definitions()

    models = [ZeroShotClassifier, LinearProbeClassifier, WeightedKNNClassifier, I2IRetrievalEvaluator]
    performances = run_evaluation(models, definitions)
    if save:
        export_evaluation_to_csv(performances)


@cli.command(
    "animate",
    help="""Animate 2D embeddings from two different models on the same dataset.
The interface will prompt you to choose which embeddings you want to use.
""",
)
def animate_embeddings(
    interactive: Annotated[bool, Option(help="Interactive plot instead of animation.")] = False,
    reduction: Annotated[str, Option(help="Reduction type [pca, tsne, umap (default)].")] = "umap",
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
    all_: Annotated[
        bool,
        Option("--all", "-a", help="List all models and datasets that are available via the tool."),
    ] = False,
):
    from clip_eval.dataset import DatasetProvider
    from clip_eval.model import ModelProvider

    if all_:
        datasets = DatasetProvider.list_dataset_titles()
        models = ModelProvider.list_model_titles()
        print(f"Available datasets are: {', '.join(datasets)}")
        print(f"Available models are: {', '.join(models)}")
        return

    defns = read_all_cached_embeddings(as_list=True)
    print(f"Available model_datasets pairs: {', '.join([str(defn) for defn in defns])}")


if __name__ == "__main__":
    cli()
