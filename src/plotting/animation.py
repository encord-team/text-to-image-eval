import pathlib

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from reduction import PCAReducer, UMAPReducer

from src.common import EmbeddingArray, EmbeddingDefinition, Embeddings


def update_plot(t):
    interpolation = reduced_1 * (1 - t) + reduced_2 * t
    scat.set_offsets(interpolation)
    return scat


if __name__ == "__main__":
    defn_1 = EmbeddingDefinition(model="clip", dataset="LungCancer4Types")
    defn_2 = EmbeddingDefinition(model="pubmed", dataset="LungCancer4Types")
    embd_1 = defn_1.load_embeddings()
    embd_2 = defn_2.load_embeddings()
    print("Embeddings loaded")
    reduced_1 = UMAPReducer.reduce(embd_1.images)
    reduced_2 = UMAPReducer.reduce(embd_2.images)
    print("Reductions made")
    fig = plt.gcf()

    scat = plt.scatter(reduced_1[:, 0], reduced_1[:, 1], c=embd_1.labels)
    # plt.scatter(reduced_2[:, 0], reduced_2[:, 1], c=embd_2.labels, facecolors="none")
    plt.legend(*scat.legend_elements(), loc="upper right")
    anim = animation.FuncAnimation(fig, update_plot, frames=np.arange(0, 1, 0.05))
    defn_str = lambda x: f"{x.model}_{x.dataset}"
    filename = f"Transition_{defn_str(defn_1)}_{defn_str(defn_2)}.gif"
    OUTPUT_DIR = pathlib.Path("./output/animations/")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH = OUTPUT_DIR / filename
    anim.save(OUTPUT_PATH)
    plt.show()
