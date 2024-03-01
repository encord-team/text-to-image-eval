from datetime import datetime
from pathlib import Path
from typing import Literal, overload

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from PIL import Image

from clip_eval.common import EmbeddingDefinition
from clip_eval.common.data_models import SafeName
from clip_eval.common.numpy_types import ClassArray, N2Array
from clip_eval.constants import OUTPUT_PATH
from clip_eval.dataset import dataset_provider

from .reduction import REDUCTIONS, reduction_from_string


@overload
def create_embedding_chart(
    x1: N2Array,
    x2: N2Array,
    labels: ClassArray,
    title1: str | SafeName,
    title2: str | SafeName,
    suptitle: str,
    label_names: list[str],
    *,
    interactive: Literal[False],
) -> animation.FuncAnimation:
    ...


@overload
def create_embedding_chart(
    x1: N2Array,
    x2: N2Array,
    labels: ClassArray,
    title1: str | SafeName,
    title2: str | SafeName,
    suptitle: str,
    label_names: list[str],
    *,
    interactive: Literal[True],
) -> None:
    ...


@overload
def create_embedding_chart(
    x1: N2Array,
    x2: N2Array,
    labels: ClassArray,
    title1: str | SafeName,
    title2: str | SafeName,
    suptitle: str,
    label_names: list[str],
    *,
    interactive: bool,
) -> animation.FuncAnimation | None:
    ...


def create_embedding_chart(
    x1: N2Array,
    x2: N2Array,
    labels: ClassArray,
    title1: str | SafeName,
    title2: str | SafeName,
    suptitle: str,
    label_names: list[str],
    *,
    interactive: bool = False,
) -> animation.FuncAnimation | None:
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(suptitle)

    # Compute chart bounds
    all = np.concatenate([x1, x2], axis=0)
    xmin, ymin = all.min(0)
    xmax, ymax = all.max(0)
    dx = xmax - xmin
    dy = ymax - ymin
    xmin -= dx * 0.1
    xmax += dx * 0.1
    ymin -= dy * 0.1
    ymax += dy * 0.1

    # Initial plot
    points = plt.scatter(*x1.T, c=labels)

    ax.axis("off")
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    handles, _ = points.legend_elements()
    ax.legend(handles, label_names, loc="upper right")

    # Position scatter and slider
    s_width, s_height = (0.4, 0.1)
    x_padding = (1 - s_width) / 2

    fig.subplots_adjust(bottom=s_height * 1.5)
    # Insert Logo
    logo = Image.open(Path(__file__).parents[2] / "images" / "encord.png")
    w, h = logo.size
    logo_padding_x = 0.02
    ax_w = x_padding - logo_padding_x * 2
    ax_h = ax_w * h / w
    ax_logo = fig.add_axes((0.02, 0.02, ax_w, ax_h))
    ax_logo.axis("off")
    ax_logo.imshow(logo)

    # Build slider
    ax_interpolation = fig.add_axes((x_padding, s_height, s_width, s_height / 2))
    init_time = 0

    def interpolate(t):
        t = max(min(t, 1), 0)
        return ((1 - t) * x1 + t * x2).T

    interpolate_slider = Slider(
        ax=ax_interpolation,
        label="",
        valmin=0.0,
        valmax=1.0,
        valinit=init_time,
    )

    # Add model_titles
    fig.text(x_padding - 0.02, s_height * 1.25, title1, ha="right", va="center")
    fig.text(1 - x_padding + 0.06, 0.1 * 1.25, title2, ha="left", va="center")

    def update_from_slider(val):
        points.set_offsets(interpolate(interpolate_slider.val).T)

    interpolate_slider.on_changed(update_from_slider)

    if interactive:
        plt.show()
        return

    # Animation bit
    frames_left = np.linspace(0, 1, 20) ** 4 / 2
    frames = np.concatenate([frames_left, 1 - frames_left[::-1][1:]], axis=0)
    frames = np.concatenate([frames, frames[::-1]], axis=0)

    def update_from_animation(val):
        interpolate_slider.set_val(val)

    interpolate_slider.set_active(False)

    return animation.FuncAnimation(fig, update_from_animation, frames=frames)  # type: ignore


def standardize(a: N2Array) -> N2Array:
    mins = a.min(0, keepdims=True)
    maxs = a.max(0, keepdims=True)
    return (a - mins) / (maxs - mins)


def rotate_to_target(source: N2Array, destination: N2Array) -> N2Array:
    from scipy.spatial.transform import Rotation as R

    source = np.pad(source, [(0, 0), (0, 1)], mode="constant", constant_values=0.0)
    destination = np.pad(destination, [(0, 0), (0, 1)], mode="constant", constant_values=0.0)

    rot, *_ = R.align_vectors(source, destination, return_sensitivity=True)
    out = source @ rot.as_matrix()
    return out[:, :2]


@overload
def build_animation(
    defn_1: EmbeddingDefinition,
    defn_2: EmbeddingDefinition,
    *,
    interactive: Literal[True],
    reduction: REDUCTIONS = "umap",
) -> None:
    ...


@overload
def build_animation(
    defn_1: EmbeddingDefinition,
    defn_2: EmbeddingDefinition,
    *,
    reduction: REDUCTIONS = "umap",
    interactive: Literal[False],
) -> animation.FuncAnimation:
    ...


@overload
def build_animation(
    defn_1: EmbeddingDefinition,
    defn_2: EmbeddingDefinition,
    *,
    reduction: REDUCTIONS = "umap",
    interactive: bool,
) -> animation.FuncAnimation | None:
    ...


def build_animation(
    defn_1: EmbeddingDefinition,
    defn_2: EmbeddingDefinition,
    *,
    reduction: REDUCTIONS = "umap",
    interactive: bool = False,
) -> animation.FuncAnimation | None:
    dataset = dataset_provider.get_dataset(defn_1.dataset)

    embeds = defn_1.load_embeddings()  # FIXME: This is expensive to get just labels
    if embeds is None:
        raise ValueError("Empty embeddings")

    reducer = reduction_from_string(reduction)
    reduced_1 = standardize(reducer.get_reduction(defn_1))
    reduced_2 = rotate_to_target(standardize(reducer.get_reduction(defn_2)), reduced_1)

    return create_embedding_chart(
        reduced_1,
        reduced_2,
        embeds.labels,
        defn_1.model,
        defn_2.model,
        suptitle=dataset.title,
        label_names=dataset.class_names,
        interactive=interactive,
    )


def save_animation_to_file(anim: animation.FuncAnimation, defn_1, defn_2: EmbeddingDefinition):
    ts = datetime.now()
    animation_file = OUTPUT_PATH.ANIMATIONS / f"transition_{defn_1}-{defn_2}_{ts.isoformat()}.gif"
    animation_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure that parent folder exists
    anim.save(animation_file)
    print(f"Stored animation in `{animation_file}`")


if __name__ == "__main__":
    defn_1 = EmbeddingDefinition(model="clip", dataset="LungCancer4Types")
    defn_2 = EmbeddingDefinition(model="pubmed", dataset="LungCancer4Types")
    anim = build_animation(defn_1, defn_2, interactive=False)
    ts = datetime.now()
    animation_file = OUTPUT_PATH.ANIMATIONS / f"transition_{defn_1}-{defn_2}_{ts.isoformat()}.gif"
    animation_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure that parent folder exists
    anim.save(animation_file)
    plt.show()
