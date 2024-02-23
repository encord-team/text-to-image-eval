import json
from pathlib import Path

from encord import Project
from encord.orm.dataset import DataType, Image, Video
from encord.project import LabelRowV2
from tqdm.auto import tqdm

from src.dataset.utils import collect_async, download_file


def download_image(image_data: Image | Video, destination_dir: Path) -> Path:
    # TODO The type of `image_data` is also Video because of a SDK bug explained in `download_label_row_data`.
    destination_path = destination_dir / image_data.title
    if not destination_path.exists():
        download_file(image_data.file_link, destination_path)
    return destination_path


def download_label_row_data(
    project: Project, label_row: LabelRowV2, data_dir: Path, overwrite_annotations: bool = False
) -> list[Path]:
    label_row_dir = data_dir / label_row.data_hash
    label_row_dir.mkdir(parents=True, exist_ok=True)

    # Download the annotations
    label_row.initialise_labels()
    label_row_annotations = label_row_dir / "annotations.json"
    if not label_row_annotations.exists() or overwrite_annotations:
        label_row_annotations.write_text(json.dumps(label_row.to_encord_dict()), encoding="utf-8")

    # Download the images
    if label_row.data_type == DataType.IMAGE:
        # TODO This `if` is here because of a SDK bug, remove it when IMAGE data is stored in the proper image field [1]
        images_data = [project.get_data(label_row.data_hash, get_signed_url=True)[0]]
    else:
        images_data = project.get_data(label_row.data_hash, get_signed_url=True)[1]
    return collect_async(
        lambda image_data: download_image(image_data, label_row_dir),
        images_data,
        max_workers=4,
        disable=True,
    )


def download_data_from_project(project: Project, data_dir: Path, overwrite_annotations: bool = False) -> None:
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
    :param overwrite_annotations: Flag that indicates whether to overwrite existing annotations if they exist.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    data_hashes = [lr_metadata.data_hash for lr_metadata in project.list_label_rows(include_uninitialised_labels=True)]
    for label_row in tqdm(project.list_label_rows_v2(), desc=f"Downloading [{project.title}]"):
        if label_row.data_type in {DataType.IMAGE, DataType.IMG_GROUP}:
            download_label_row_data(project, label_row, data_dir, overwrite_annotations)

    with tqdm(total=len(data_hashes), desc=f"Downloading [{project.title}]") as pbar:
        for data_hash in data_hashes:
            download_label_row_data(project, data_hash, data_dir, overwrite_annotations)
            pbar.update(1)
