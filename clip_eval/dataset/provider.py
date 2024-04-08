from copy import deepcopy
from typing import Any

from natsort import natsorted, ns

from clip_eval.constants import CACHE_PATH

from .base import Dataset, Split
from .encord import EncordDataset
from .hf import HFDataset


class DatasetProvider:
    def __init__(self):
        self._datasets = {}
        self._global_settings: dict[str, Any] = dict()

    @property
    def global_settings(self) -> dict:
        return deepcopy(self._global_settings)

    def add_global_setting(self, name: str, value: Any) -> None:
        self._global_settings[name] = value

    def remove_global_setting(self, name: str):
        self._global_settings.pop(name, None)

    def register_dataset(self, source: type[Dataset], title: str, split: Split | None = None, **kwargs):
        if split is None:
            # One dataset with all the split definitions
            self._datasets[title] = (source, kwargs)
        else:
            # One dataset is defined per split
            kwargs.update(split=split)
            self._datasets[(title, split)] = (source, kwargs)

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
