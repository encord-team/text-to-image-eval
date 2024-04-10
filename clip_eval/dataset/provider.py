import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from natsort import natsorted, ns

from clip_eval.constants import CACHE_PATH

from .base import Dataset, Split
from .encord import EncordDataset
from .hf import HFDataset
from .utils import load_class_from_path


class DatasetProvider:
    def __init__(self):
        self._datasets = {}
        self._global_settings: dict[str, Any] = dict()
        self.__known_dataset_types: dict[tuple[str, str], Any] = dict()

    @property
    def global_settings(self) -> dict:
        return deepcopy(self._global_settings)

    def add_global_setting(self, name: str, value: Any) -> None:
        self._global_settings[name] = value

    def remove_global_setting(self, name: str) -> None:
        self._global_settings.pop(name, None)

    def register_dataset(self, source: type[Dataset], title: str, split: Split | None = None, **kwargs) -> None:
        if split is None:
            # One dataset with all the split definitions
            self._datasets[title] = (source, kwargs)
        else:
            # One dataset is defined per split
            kwargs.update(split=split)
            self._datasets[(title, split)] = (source, kwargs)

    def register_dataset_from_json_definition(self, json_definition: Path) -> None:
        dataset_params: dict = json.loads(json_definition.read_text(encoding="utf-8"))
        module_path: str | None = dataset_params.pop("module_path", None)
        dataset_type_name: str | None = dataset_params.pop("dataset_type", None)
        if module_path is None or dataset_type_name is None:
            raise ValueError(
                f"Missing required fields `module_path` or `dataset_type` in "
                f"the JSON definition file: {json_definition.as_posix()}"
            )

        # Fetch the class of the dataset type stated in the definition
        dataset_type = self.__known_dataset_types.get((module_path, dataset_type_name))
        if dataset_type is None:
            dataset_type = load_class_from_path(module_path, dataset_type_name)
            if not issubclass(dataset_type, Dataset):
                raise ValueError(
                    f"Dataset type specified in the JSON definition file `{json_definition.as_posix()}` "
                    f"does not inherit from the base class `Dataset`"
                )
            self.__known_dataset_types[(module_path, dataset_type_name)] = dataset_type
        self.register_dataset(dataset_type, **dataset_params)

    def register_datasets_from_sources_dir(self, source_dir: Path) -> None:
        for f in source_dir.glob("*.json"):
            self.register_dataset_from_json_definition(f)

    def get_dataset(self, title: str, split: Split) -> Dataset:
        if (title, split) in self._datasets:
            # The split corresponds to fetching a whole dataset (one-to-one relationship)
            dict_key = (title, split)
            split = Split.ALL  # Ensure to read the whole dataset
        elif title in self._datasets:
            # The dataset internally knows how to determine the split
            dict_key = title
        else:
            raise ValueError(f"Unrecognized dataset: {title}")

        source, kwargs = self._datasets[dict_key]
        # Apply global settings. Values of local settings take priority when local and global settings share keys.
        kwargs_with_global_settings = self.global_settings | kwargs
        return source(title, split=split, **kwargs_with_global_settings)

    def list_dataset_titles(self) -> list[str]:
        dataset_titles = [
            dict_key[0] if isinstance(dict_key, tuple) else dict_key for dict_key in self._datasets.keys()
        ]
        return natsorted(set(dataset_titles), alg=ns.IGNORECASE)


dataset_provider = DatasetProvider()
# Global settings
dataset_provider.add_global_setting("cache_dir", CACHE_PATH)

# Hugging Face datasets
dataset_provider.register_dataset(HFDataset, "plants", title_in_source="sampath017/plants")
dataset_provider.register_dataset(HFDataset, "Alzheimer-MRI", title_in_source="Falah/Alzheimer_MRI")
dataset_provider.register_dataset(HFDataset, "skin-cancer", title_in_source="marmal88/skin_cancer", target_feature="dx")
dataset_provider.register_dataset(HFDataset, "geo-landmarks", title_in_source="Qdrant/google-landmark-geo")
dataset_provider.register_dataset(
    HFDataset,
    "LungCancer4Types",
    title_in_source="Kabil007/LungCancer4Types",
    revision="a1aab924c6bed6b080fc85552fd7b39724931605",
)
dataset_provider.register_dataset(
    HFDataset,
    "NIH-Chest-X-ray",
    title_in_source="alkzar90/NIH-Chest-X-ray-dataset",
    name="image-classification",
    target_feature="labels",
    trust_remote_code=True,
)
dataset_provider.register_dataset(
    HFDataset,
    "chest-xray-classification",
    title_in_source="trpakov/chest-xray-classification",
    name="full",
    target_feature="labels",
)

dataset_provider.register_dataset(
    HFDataset,
    "sports-classification",
    title_in_source="HES-XPLAIN/SportsImageClassification",
)

# Encord datasets
dataset_provider.register_dataset(
    EncordDataset,
    "rsicd",
    project_hash="46ba913e-1428-48ef-be7f-2553e69bc1e6",
    classification_hash="4f6cf0c8",
)
