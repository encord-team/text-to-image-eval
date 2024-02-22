from collections.abc import Callable
from datetime import datetime

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection

from clip_eval.common import EmbeddingDefinition
from clip_eval.common.numpy_types import N2Array
from clip_eval.constants import OUTPUT_PATH
from clip_eval.dataset import dataset_provider

from .reduction import REDUCTIONS, Reducer, reduction_from_string


def build_update_plot(scat: PathCollection, reduced_1, reduced_2: N2Array) -> Callable[[float], PathCollection]:
    def update_plot(t: float) -> PathCollection:
        interpolation = reduced_1 * (1 - t) + reduced_2 * t
        scat.set_offsets(interpolation)
        return scat

    return update_plot


def lazy_reduce(d: EmbeddingDefinition, reduction: Reducer) -> N2Array:
    red_file = d.get_reduction_path(reduction.title())
    if red_file.is_file():
        reduced = np.load(red_file)
    else:
        reduced = reduction.get_reduction(d)
        np.save(red_file, reduced)
    return reduced


def standardize(a: N2Array):
    mins = a.min(0, keepdims=True)
    maxs = a.max(0, keepdims=True)
    return (a - mins) / (maxs - mins)


def rotate_to_target(source: N2Array, destination: N2Array):
    from scipy.spatial.transform import Rotation as R

    source = np.pad(source, [(0, 0), (0, 1)], mode="constant", constant_values=0.0)
    destination = np.pad(destination, [(0, 0), (0, 1)], mode="constant", constant_values=0.0)

    rot, *_ = R.align_vectors(source, destination, return_sensitivity=True)
    out = source @ rot.as_matrix()
    print(out[:, 2].std())
    return out[:, :2]


def build_animation(
    defn_1: EmbeddingDefinition,
    defn_2: EmbeddingDefinition,
    reduction: REDUCTIONS = "umap",
) -> animation.FuncAnimation:
    dataset = dataset_provider.get_dataset(defn_1.dataset)
    label_names: list[str] = dataset._dataset.info.features["label"].names  # label is hardcoded here

    embeds = defn_1.load_embeddings()  # FIXME: This is expensive to get just labels
    if embeds is None:
        raise ValueError("Empty embeddings")
    labels = embeds.labels

    reducer = reduction_from_string(reduction)

    reduced_1 = standardize(lazy_reduce(defn_1, reducer))
    reduced_2 = rotate_to_target(standardize(lazy_reduce(defn_2, reducer)), reduced_1)

    print("Reductions made")

    fig = plt.figure(figsize=(10, 10))
    scat: PathCollection = plt.scatter(*reduced_1.T, c=labels)
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    update_plot = build_update_plot(scat, reduced_1, reduced_2)
    handles, labels = scat.legend_elements()
    labels = label_names
    plt.legend(handles, labels, loc="upper right")
    frames = np.concatenate([np.arange(0, 1, 0.01), np.arange(0, 1, 0.05)[::-1]], axis=0)
    anim = animation.FuncAnimation(fig, update_plot, frames=frames)
    plt.title(f"Dataset: {defn_1.dataset}, Transition: {defn_1.model} to {defn_2.model}")
    return anim


def save_animation_to_file(anim: animation.FuncAnimation, defn_1, defn_2: EmbeddingDefinition):
    ts = datetime.now()
    animation_file = OUTPUT_PATH.ANIMATIONS / f"transition_{defn_1}-{defn_2}_{ts.isoformat()}.gif"
    animation_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure that parent folder exists
    anim.save(animation_file)


if __name__ == "__main__":
    defn_1 = EmbeddingDefinition(model="clip", dataset="LungCancer4Types")
    defn_2 = EmbeddingDefinition(model="pubmed", dataset="LungCancer4Types")
    anim = build_animation(defn_1, defn_2)
    ts = datetime.now()
    animation_file = OUTPUT_PATH.ANIMATIONS / f"transition_{defn_1}-{defn_2}_{ts.isoformat()}.gif"
    animation_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure that parent folder exists
    anim.save(animation_file)
    plt.show()
