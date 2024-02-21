from collections.abc import Callable
from datetime import datetime

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection

from src.common import EmbeddingDefinition
from src.common.numpy_types import N2Array
from src.constants import OUTPUT_PATH
from src.dataset import dataset_provider
from src.plotting.reduction import UMAPReducer


def build_update_plot(scat: PathCollection, reduced_1, reduced_2: N2Array) -> Callable[[float], PathCollection]:
    def update_plot(t: float) -> PathCollection:
        interpolation = reduced_1 * (1 - t) + reduced_2 * t
        scat.set_offsets(interpolation)
        return scat

    return update_plot


def build_animation(defn_1: EmbeddingDefinition, defn_2: EmbeddingDefinition) -> animation.FuncAnimation:
    dataset_1 = dataset_provider.get_dataset(defn_1.dataset)
    label_names: list[str] = dataset_1._dataset.info.features["label"].names  # label is hardcoded here

    embd_1 = defn_1.load_embeddings()  # Needed for obtaining the labels for the plot
    reduced_1 = UMAPReducer.get_reduction(defn_1)
    reduced_2 = UMAPReducer.get_reduction(defn_2)
    print("Reductions made")
    fig = plt.figure(figsize=(10, 10))

    scat: PathCollection = plt.scatter(reduced_1[:, 0], reduced_1[:, 1], c=embd_1.labels)
    update_plot = build_update_plot(scat, reduced_1, reduced_2)
    handles, labels = scat.legend_elements()
    labels = label_names
    plt.legend(handles, labels, loc="upper right")
    anim = animation.FuncAnimation(fig, update_plot, frames=np.arange(0, 1, 0.05))
    plt.title(f"Transition: {defn_1} to {defn_2}")
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
