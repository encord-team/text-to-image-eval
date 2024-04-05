import json
from pathlib import Path

from encord import Project
from encord.orm.dataset import DataType, Image, Video
from encord.project import LabelRowV2
from tqdm.auto import tqdm

from .utils import Split, collect_async, download_file, simple_random_split


def download_image(image_data: Image | Video, destination_dir: Path) -> Path:
    # TODO The type of `image_data` is also Video because of a SDK bug explained in `download_label_row_data`.
    file_name = get_frame_name(image_data.image_hash, image_data.title)
    destination_path = destination_dir / file_name
    if not destination_path.exists():
        download_file(image_data.file_link, destination_path)
    return destination_path


def download_label_row_data(
    data_dir: Path, project: Project, label_row: LabelRowV2, overwrite_annotations: bool = False
) -> list[Path]:
    label_row_annotations = get_label_row_annotations_file(data_dir, project.project_hash, label_row.label_hash)
    label_row_dir = label_row_annotations.parent
    label_row_dir.mkdir(parents=True, exist_ok=True)

    # Download the annotations
    if not label_row_annotations.exists() or overwrite_annotations:
        label_row.initialise_labels()
        label_row_annotations.write_text(json.dumps(label_row.to_encord_dict()), encoding="utf-8")
    else:
        # Needed to iterate the label row's frames
        label_row.from_labels_dict(json.loads(label_row_annotations.read_text(encoding="utf-8")))

    # Download the images
    is_frame_missing = any(
        not get_frame_file(data_dir, project.project_hash, label_row, idx).exists()
        for idx in range(label_row.number_of_frames)
    )
    if not is_frame_missing:
        return [label_row_dir / fv.image_title for fv in label_row.get_frame_views()]

    if label_row.data_type == DataType.IMAGE:
        # TODO This `if` is here because of a SDK bug, remove it when IMAGE data is stored in the proper image field [1]
        images_data = [project.get_data(label_row.data_hash, get_signed_url=True)[0]]
        # Missing field caused by the SDK bug
        images_data[0]["image_hash"] = label_row.data_hash
    else:
        images_data = project.get_data(label_row.data_hash, get_signed_url=True)[1]
    return collect_async(
        lambda image_data: download_image(image_data, label_row_dir),
        images_data,
        max_workers=4,
        disable=True,
    )


def download_data_from_project(
    project: Project,
    data_dir: Path,
    label_hashes: list[str] | None = None,
    overwrite_annotations: bool = False,
) -> None:
    """
    Iterates through the images of the project and downloads their content, adhering to the following file structure:
    data_dir/
    ├── project-hash/
    │   ├── image-group-hash1/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   │   └── annotations.json
    │   └── image-hash1/
    │   │   ├── image1.jpeg
    │   │   ├── annotations.json
    │   └── ...
    └── ...
    :param project: The project containing the images with their annotations.
    :param data_dir: The directory where the project data will be downloaded.
    :param label_hashes: The hashes of the label rows that will be downloaded. If None, all label rows
        will be downloaded.
    :param overwrite_annotations: Flag that indicates whether to overwrite existing annotations if they exist.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    for label_row in tqdm(
        project.list_label_rows_v2(label_hashes=label_hashes),
        desc=f"Fetching data from Encord project `{project.title}`",
    ):
        if label_row.data_type in {DataType.IMAGE, DataType.IMG_GROUP}:
            download_label_row_data(data_dir, project, label_row, overwrite_annotations)


def get_frame_name(frame_hash: str, frame_title: str) -> str:
    file_extension = frame_title.rsplit(sep=".", maxsplit=1)[-1]
    return f"{frame_hash}.{file_extension}"


def get_frame_file(data_dir: Path, project_hash: str, label_row: LabelRowV2, frame: int) -> Path:
    label_row_dir = get_label_row_dir(data_dir, project_hash, label_row.label_hash)
    frame_view = label_row.get_frame_view(frame)
    return label_row_dir / get_frame_name(frame_view.image_hash, frame_view.image_title)


def get_frame_file_raw(
    data_dir: Path, project_hash: str, label_row_hash: str, frame_hash: str, frame_title: str
) -> Path:
    return get_label_row_dir(data_dir, project_hash, label_row_hash) / get_frame_name(frame_hash, frame_title)


def get_label_row_annotations_file(data_dir: Path, project_hash: str, label_row_hash: str) -> Path:
    return get_label_row_dir(data_dir, project_hash, label_row_hash) / "annotations.json"


def get_label_row_dir(data_dir: Path, project_hash: str, label_row_hash: str) -> Path:
    return data_dir / project_hash / label_row_hash


def simple_project_split(
    project: Project,
    seed: int = 42,
    train_split: float = 0.7,
    validation_split: float = 0.15,
) -> dict[Split, list[str]]:
    """
    Split the label rows of a project into training, validation, and test sets using simple random splitting.

    :param project: The project containing the label rows to split.
    :param seed: Random seed for reproducibility. Defaults to 42.
    :param train_split: Percentage of the dataset to allocate to the training set. Defaults to 0.7.
    :param validation_split: Percentage of the dataset to allocate to the validation set. Defaults to 0.15.
    :return: A dictionary containing lists with the label hashes of the data represented in the training,
        validation, and test sets.

    :raises ValueError: If the sum of `train_split` and `validation_split` is greater than 1,
        or if `train_split` or `validation_split` are less than 0.
    """
    label_rows = project.list_label_rows_v2()
    split_to_indices = simple_random_split(len(label_rows), seed, train_split, validation_split)
    return {split: [label_rows[i].label_hash for i in indices] for split, indices in split_to_indices.items()}
