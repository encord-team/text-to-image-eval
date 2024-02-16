from datetime import datetime

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from src.common import EmbeddingDefinition
from src.constants import OUTPUT_PATH
from src.plotting.reduction import UMAPReducer


def update_plot(t):
    interpolation = reduced_1 * (1 - t) + reduced_2 * t
    scat.set_offsets(interpolation)
    return scat


if __name__ == "__main__":
    defn_1 = EmbeddingDefinition(model="clip", dataset="LungCancer4Types")
    defn_2 = EmbeddingDefinition(model="pubmed", dataset="LungCancer4Types")

    embd_1 = defn_1.load_embeddings()  # Needed for obtaining the labels for the plot
    reduced_1 = UMAPReducer.get_reduction(defn_1)
    reduced_2 = UMAPReducer.get_reduction(defn_2)
    print("Reductions made")
    fig = plt.gcf()

    scat = plt.scatter(reduced_1[:, 0], reduced_1[:, 1], c=embd_1.labels)
    # plt.scatter(reduced_2[:, 0], reduced_2[:, 1], c=embd_2.labels, facecolors="none")
    plt.legend(*scat.legend_elements(), loc="upper right")
    anim = animation.FuncAnimation(fig, update_plot, frames=np.arange(0, 1, 0.05))
    ts = datetime.now()
    animation_file = OUTPUT_PATH.ANIMATIONS / f"transition_{defn_1}-{defn_2}_{ts.isoformat()}.gif"
    animation_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure that parent folder exists
    anim.save(animation_file)
    plt.show()
