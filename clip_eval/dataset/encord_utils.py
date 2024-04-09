import json
from pathlib import Path

from encord import Project
from encord.common.constants import DATETIME_STRING_FORMAT
from encord.orm.dataset import DataType, Image, Video
from encord.project import LabelRowV2
from tqdm.auto import tqdm

from .utils import Split, collect_async, download_file, simple_random_split


def _download_image(image_data: Image | Video, destination_dir: Path) -> Path:
    # TODO The type of `image_data` is also Video because of a SDK bug explained in `_download_label_row_image_data`.
    file_name = get_frame_name(image_data.image_hash, image_data.title)
    destination_path = destination_dir / file_name
    if not destination_path.exists():
        download_file(image_data.file_link, destination_path)
    return destination_path


def _download_label_row_image_data(data_dir: Path, project: Project, label_row: LabelRowV2) -> list[Path]:
    label_row.initialise_labels()
    label_row_dir = get_label_row_dir(data_dir, project.project_hash, label_row.label_hash)
    label_row_dir.mkdir(parents=True, exist_ok=True)

    if label_row.data_type == DataType.IMAGE:
        # TODO This `if` is here because of a SDK bug, remove it when IMAGE data is stored in the proper image field [1]
        images_data = [project.get_data(label_row.data_hash, get_signed_url=True)[0]]
        # Missing field caused by the SDK bug
        images_data[0]["image_hash"] = label_row.data_hash
    else:
        images_data = project.get_data(label_row.data_hash, get_signed_url=True)[1]
    return collect_async(
        lambda image_data: _download_image(image_data, label_row_dir),
        images_data,
        max_workers=4,
        disable=True,
    )


def _download_label_rows(
    project: Project,
    data_dir: Path,
    label_rows: list[LabelRowV2],
    overwrite_annotations: bool,
    downloaded_label_rows_tracker: dict,
    tqdm_desc: str | None,
):
    if tqdm_desc is None:
        tqdm_desc = f"Downloading data from Encord project `{project.title}`"

    for label_row in tqdm(label_rows, desc=tqdm_desc):
        if label_row.data_type not in {DataType.IMAGE, DataType.IMG_GROUP}:
            continue
        save_annotations = False
        # Trigger the images download if the label hash is not found or is None (never downloaded).
        if label_row.label_hash not in downloaded_label_rows_tracker.keys():
            _download_label_row_image_data(data_dir, project, label_row)
            save_annotations = True
        # Overwrite annotations only if `last_edited_at` values differ between the existing and new annotations.
        elif (
            overwrite_annotations
            and label_row.last_edited_at.strftime(DATETIME_STRING_FORMAT)
            != downloaded_label_rows_tracker[label_row.label_hash]["last_edited_at"]
        ):
            label_row.initialise_labels()
            save_annotations = True

        if save_annotations:
            annotations_file = get_label_row_annotations_file(data_dir, project.project_hash, label_row.label_hash)
            annotations_file.write_text(json.dumps(label_row.to_encord_dict()), encoding="utf-8")
            downloaded_label_rows_tracker[label_row.label_hash] = {"last_edited_at": label_row.last_edited_at}


def download_data_from_project(
    project: Project,
    data_dir: Path,
    label_rows: list[LabelRowV2] | None = None,
    overwrite_annotations: bool = False,
    tqdm_desc: str | None = None,
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
    :param label_rows: The label rows that will be downloaded. If None, all label rows will be downloaded.
    :param overwrite_annotations: Flag that indicates whether to overwrite existing annotations if they exist.
    :param tqdm_desc: Optional description for tqdm progress bar.
        Defaults to 'Downloading data from Encord project `{project.title}`'
    """
    # Read internal file that controls the downloaded data progress
    downloaded_label_rows_tracker_file = data_dir / project.project_hash / "label_rows.json"
    downloaded_label_rows_tracker_file.parent.mkdir(parents=True, exist_ok=True)
    downloaded_label_rows_tracker = (
        json.loads(downloaded_label_rows_tracker_file.read_text(encoding="utf-8"))
        if downloaded_label_rows_tracker_file.is_file()
        else dict()
    )

    # Retrieve only the unseen data if there is no explicit annotation update
    filtered_label_rows = (
        label_rows
        if overwrite_annotations
        else [lr for lr in label_rows if lr.label_hash not in downloaded_label_rows_tracker.keys()]
    )
    if len(filtered_label_rows) == 0:
        return

    try:
        _download_label_rows(
            project,
            data_dir,
            filtered_label_rows,
            overwrite_annotations,
            downloaded_label_rows_tracker,
            tqdm_desc=tqdm_desc,
        )
    finally:
        # Save the current download progress in case of failure
        downloaded_label_rows_tracker_file.write_text(json.dumps(downloaded_label_rows_tracker), encoding="utf-8")


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
    enforce_label_rows_initialization(label_rows)  # Ensure that all label rows have a label hash
    return {split: [label_rows[i].label_hash for i in indices] for split, indices in split_to_indices.items()}


def enforce_label_rows_initialization(label_rows: list[LabelRowV2]):
    for lr in label_rows:
        if lr.label_hash is None:
            lr.initialise_labels()
