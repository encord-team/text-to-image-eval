import json
from pathlib import Path

from encord import Project
from encord.orm.dataset import DataType, Image, Video
from encord.project import LabelRowV2
from tqdm.auto import tqdm

from .utils import collect_async, download_file


def download_image(image_data: Image | Video, destination_dir: Path) -> Path:
    # TODO The type of `image_data` is also Video because of a SDK bug explained in `download_label_row_data`.
    destination_path = destination_dir / image_data.title
    if not destination_path.exists():
        download_file(image_data.file_link, destination_path)
    return destination_path


def download_label_row_data(
    project: Project, label_row: LabelRowV2, data_dir: Path, overwrite_annotations: bool = False
) -> list[Path]:
    label_row_dir = data_dir / project.project_hash / label_row.data_hash
    label_row_dir.mkdir(parents=True, exist_ok=True)

    # Download the annotations
    label_row_annotations = label_row_dir / "annotations.json"
    if not label_row_annotations.exists() or overwrite_annotations:
        label_row.initialise_labels()
        label_row_annotations.write_text(json.dumps(label_row.to_encord_dict()), encoding="utf-8")
    else:
        # Needed to iterate the label row's frames
        label_row.from_labels_dict(json.loads(label_row_annotations.read_text(encoding="utf-8")))

    # Download the images
    is_frame_missing = any(not (label_row_dir / fv.image_title).exists() for fv in label_row.get_frame_views())
    if not is_frame_missing:
        return [label_row_dir / fv.image_title for fv in label_row.get_frame_views()]

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
    for label_row in tqdm(project.list_label_rows_v2(), desc=f"Downloading [{project.title}]"):
        if label_row.data_type in {DataType.IMAGE, DataType.IMG_GROUP}:
            download_label_row_data(project, label_row, data_dir, overwrite_annotations)
