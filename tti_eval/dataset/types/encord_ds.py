import json
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

from encord import EncordUserClient, Project
from encord.common.constants import DATETIME_STRING_FORMAT
from encord.objects import Classification, LabelRowV2
from encord.objects.common import PropertyType
from encord.orm.dataset import DataType, Image, Video
from PIL import Image as PILImage
from tqdm.auto import tqdm

from tti_eval.dataset import Dataset, Split
from tti_eval.dataset.utils import collect_async, download_file, simple_random_split


class EncordDataset(Dataset):
    def __init__(
        self,
        title: str,
        project_hash: str,
        classification_hash: str,
        *,
        split: Split = Split.ALL,
        title_in_source: str | None = None,
        transform=None,
        cache_dir: str | None = None,
        ssh_key_path: str | None = None,
        **kwargs,
    ):
        super().__init__(
            title,
            split=split,
            title_in_source=title_in_source,
            transform=transform,
            cache_dir=cache_dir,
        )
        self._setup(project_hash, classification_hash, ssh_key_path, **kwargs)

    def __getitem__(self, idx):
        frame_path = self._dataset_indices_info[idx].image_file
        img = PILImage.open(frame_path)
        label = self._dataset_indices_info[idx].label

        if self.transform is not None:
            _d = self.transform(dict(image=[img], label=[label]))
            res_item = dict(image=_d["image"][0], label=_d["label"][0])
        else:
            res_item = dict(image=img, label=label)
        return res_item

    def __len__(self):
        return len(self._dataset_indices_info)

    def _get_frame_file(self, label_row: LabelRowV2, frame: int) -> Path:
        return get_frame_file(
            data_dir=self._cache_dir,
            project_hash=self._project.project_hash,
            label_row=label_row,
            frame=frame,
        )

    def _get_label_row_annotations_file(self, label_row: LabelRowV2) -> Path:
        return get_label_row_annotations_file(
            data_dir=self._cache_dir,
            project_hash=self._project.project_hash,
            label_row_hash=label_row.label_hash,
        )

    def _ensure_answers_availability(self) -> dict:
        lrs_info_file = get_label_rows_info_file(self._cache_dir, self._project.project_hash)
        label_rows_info: dict = json.loads(lrs_info_file.read_text(encoding="utf-8"))
        should_update_info = False
        class_name_to_idx = {name: idx for idx, name in enumerate(self.class_names)}  # Fast lookup of class indices
        for label_row in self._label_rows:
            if "answers" not in label_rows_info[label_row.label_hash]:
                if not label_row.is_labelling_initialised:
                    # Retrieve label row content from local storage
                    anns_path = self._get_label_row_annotations_file(label_row)
                    label_row.from_labels_dict(json.loads(anns_path.read_text(encoding="utf-8")))

                answers = dict()
                for frame_view in label_row.get_frame_views():
                    clf_instances = frame_view.get_classification_instances(self._classification)
                    # Skip frames where the input classification is missing
                    if len(clf_instances) == 0:
                        continue

                    clf_instance = clf_instances[0]
                    clf_answer = clf_instance.get_answer(self._attribute)
                    # Skip frames where the input classification has no answer (probable annotation error)
                    if clf_answer is None:
                        continue

                    answers[frame_view.frame] = {
                        "image_file": self._get_frame_file(label_row, frame_view.frame).as_posix(),
                        "label": class_name_to_idx[clf_answer.title],
                    }
                label_rows_info[label_row.label_hash]["answers"] = answers
                should_update_info = True
        if should_update_info:
            lrs_info_file.write_text(json.dumps(label_rows_info), encoding="utf-8")
        return label_rows_info

    def _setup(
        self,
        project_hash: str,
        classification_hash: str,
        ssh_key_path: str | None = None,
        **kwargs,
    ):
        ssh_key_path = ssh_key_path or os.getenv("ENCORD_SSH_KEY_PATH")
        if ssh_key_path is None:
            raise ValueError(
                "The `ssh_key_path` parameter and the `ENCORD_SSH_KEY_PATH` environment variable are both missing. "
                "Please set one of them to proceed."
            )
        client = EncordUserClient.create_with_ssh_private_key(ssh_private_key_path=ssh_key_path)
        self._project = client.get_project(project_hash)

        self._classification = self._project.ontology_structure.get_child_by_hash(
            classification_hash, type_=Classification
        )
        radio_attribute = self._classification.attributes[0]
        if radio_attribute.get_property_type() != PropertyType.RADIO:
            raise ValueError("Expected a classification hash with an attribute of type `Radio`")
        self._attribute = radio_attribute
        self.class_names = [o.title for o in self._attribute.options]

        # Fetch the label rows of the selected split
        splits_file = self._cache_dir / "splits.json"
        split_to_lr_hashes: dict[str, list[str]]
        if splits_file.exists():
            split_to_lr_hashes = json.loads(splits_file.read_text(encoding="utf-8"))
        else:
            split_to_lr_hashes = simple_project_split(self._project)
            splits_file.parent.mkdir(parents=True, exist_ok=True)
            splits_file.write_text(json.dumps(split_to_lr_hashes), encoding="utf-8")
        self._label_rows = self._project.list_label_rows_v2(label_hashes=split_to_lr_hashes[self.split])

        # Get data from source. Users may supply the `overwrite_annotations` keyword in the init to download everything
        download_data_from_project(
            self._project,
            self._cache_dir,
            self._label_rows,
            tqdm_desc=f"Downloading {self.split} data from Encord project `{self._project.title}`",
            **kwargs,
        )

        # Prepare data for the __getitem__ method
        self._dataset_indices_info: list[EncordDataset.DatasetIndexInfo] = []
        label_rows_info = self._ensure_answers_availability()
        for label_row in self._label_rows:
            answers: dict[int, Any] = label_rows_info[label_row.label_hash]["answers"]
            for frame_num in sorted(answers.keys()):
                self._dataset_indices_info.append(EncordDataset.DatasetIndexInfo(**answers[frame_num]))

    @dataclass
    class DatasetIndexInfo:
        image_file: Path | str
        label: int


# -----------------------------------------------------------------------
#                               UTILITY FUNCTIONS
# -----------------------------------------------------------------------


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


def _download_label_row(
    label_row: LabelRowV2,
    project: Project,
    data_dir: Path,
    overwrite_annotations: bool,
    label_rows_info: dict[str, Any],
    update_pbar: Callable[[], Any],
):
    if label_row.data_type not in {DataType.IMAGE, DataType.IMG_GROUP}:
        return
    save_annotations = False
    # Trigger the images download if the label hash is not found or is None (never downloaded).
    if label_row.label_hash not in label_rows_info.keys():
        _download_label_row_image_data(data_dir, project, label_row)
        save_annotations = True
    # Overwrite annotations only if `last_edited_at` values differ between the existing and new annotations.
    elif (
        overwrite_annotations
        and label_row.last_edited_at.strftime(DATETIME_STRING_FORMAT)
        != label_rows_info[label_row.label_hash]["last_edited_at"]
    ):
        label_row.initialise_labels()
        save_annotations = True

    if save_annotations:
        annotations_file = get_label_row_annotations_file(data_dir, project.project_hash, label_row.label_hash)
        annotations_file.write_text(json.dumps(label_row.to_encord_dict()), encoding="utf-8")
        label_rows_info[label_row.label_hash] = {"last_edited_at": label_row.last_edited_at}
    update_pbar()


def _download_label_rows(
    project: Project,
    data_dir: Path,
    label_rows: list[LabelRowV2],
    overwrite_annotations: bool,
    label_rows_info: dict[str, Any],
    tqdm_desc: str | None,
):
    if tqdm_desc is None:
        tqdm_desc = f"Downloading data from Encord project `{project.title}`"

    pbar = tqdm(total=len(label_rows), desc=tqdm_desc)
    _do_download = partial(
        _download_label_row,
        project=project,
        data_dir=data_dir,
        overwrite_annotations=overwrite_annotations,
        label_rows_info=label_rows_info,
        update_pbar=lambda: pbar.update(1),
    )

    with ThreadPoolExecutor(min(multiprocessing.cpu_count(), 24)) as exe:
        exe.map(_do_download, label_rows)


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
    # Read file that tracks the downloaded data progress
    lrs_info_file = get_label_rows_info_file(data_dir, project.project_hash)
    lrs_info_file.parent.mkdir(parents=True, exist_ok=True)
    label_rows_info = json.loads(lrs_info_file.read_text(encoding="utf-8")) if lrs_info_file.is_file() else dict()

    # Retrieve only the unseen data if there is no explicit annotation update
    filtered_label_rows = (
        label_rows
        if overwrite_annotations
        else [lr for lr in label_rows if lr.label_hash not in label_rows_info.keys()]
    )
    if len(filtered_label_rows) == 0:
        return

    try:
        _download_label_rows(
            project,
            data_dir,
            filtered_label_rows,
            overwrite_annotations,
            label_rows_info,
            tqdm_desc=tqdm_desc,
        )
    finally:
        # Save the current download progress in case of failure
        lrs_info_file.write_text(json.dumps(label_rows_info), encoding="utf-8")


def get_frame_name(frame_hash: str, frame_title: str) -> str:
    file_extension = frame_title.rsplit(sep=".", maxsplit=1)[-1]
    return f"{frame_hash}.{file_extension}"


def get_frame_file(data_dir: Path, project_hash: str, label_row: LabelRowV2, frame: int) -> Path:
    label_row_dir = get_label_row_dir(data_dir, project_hash, label_row.label_hash)
    frame_view = label_row.get_frame_view(frame)
    return label_row_dir / get_frame_name(frame_view.image_hash, frame_view.image_title)


def get_frame_file_raw(
    data_dir: Path,
    project_hash: str,
    label_row_hash: str,
    frame_hash: str,
    frame_title: str,
) -> Path:
    return get_label_row_dir(data_dir, project_hash, label_row_hash) / get_frame_name(frame_hash, frame_title)


def get_label_row_annotations_file(data_dir: Path, project_hash: str, label_row_hash: str) -> Path:
    return get_label_row_dir(data_dir, project_hash, label_row_hash) / "annotations.json"


def get_label_row_dir(data_dir: Path, project_hash: str, label_row_hash: str) -> Path:
    return data_dir / project_hash / label_row_hash


def get_label_rows_info_file(data_dir: Path, project_hash: str) -> Path:
    return data_dir / project_hash / "label_rows_info.json"


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
