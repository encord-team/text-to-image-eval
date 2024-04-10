import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from encord import EncordUserClient
from encord.objects import Classification, LabelRowV2
from encord.objects.common import PropertyType
from PIL import Image

from clip_eval.dataset.base import Dataset, Split
from clip_eval.dataset.encord_utils import (
    download_data_from_project,
    get_frame_file,
    get_label_row_annotations_file,
    get_label_rows_info_file,
    simple_project_split,
)


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
        super().__init__(title, split=split, title_in_source=title_in_source, transform=transform, cache_dir=cache_dir)
        self._setup(project_hash, classification_hash, ssh_key_path, **kwargs)

    def __getitem__(self, idx):
        frame_path = self._dataset_indices_info[idx].image_file
        img = Image.open(frame_path)
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
                "The `ssh_key_path` parameter and the `ENCORD_SSH_KEY_PATH` environment variable are both missing."
                "Please set one of them to proceed"
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
