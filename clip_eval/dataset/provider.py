from .dataset import Dataset, HFDataset
from .encord_dataset import EncordDataset
from typing import Any
from copy import deepcopy
from clip_eval.constants import _CACHE_PATH


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

    def register_dataset(self, title: str, source: type[Dataset], **kwargs):
        self._datasets[title] = (source, kwargs)

    def get_dataset(self, title: str) -> Dataset:
        if title not in self._datasets:
            raise ValueError(f"Unrecognized dataset: {title}")
        source, kwargs = self._datasets[title]
        # Apply global settings. Values of local settings take priority when local and global settings share keys.
        kwargs_with_global_settings = self.global_settings | kwargs
        return source(title, **kwargs_with_global_settings)

    def list_dataset_names(self) -> list[str]:
        return list(self._datasets.keys())


dataset_provider = DatasetProvider()
# Add global settings
dataset_provider.add_global_setting("cache_dir", _CACHE_PATH)

dataset_provider.register_dataset("LungCancer4Types", HFDataset, title_in_source="Kabil007/LungCancer4Types")
dataset_provider.register_dataset("plants", HFDataset, title_in_source="sampath017/plants")
dataset_provider.register_dataset(
    "NIH-Chest-X-ray",
    HFDataset,
    title_in_source="alkzar90/NIH-Chest-X-ray-dataset",
    name="image-classification",
    trust_remote_code=True,
)
dataset_provider.register_dataset("Alzheimer-MRI", HFDataset, title_in_source="Falah/Alzheimer_MRI")
dataset_provider.register_dataset("Skin-cancer", HFDataset, title_in_source="marmal88/skin_cancer")
dataset_provider.register_dataset(
    "chest-xray-classification",
    HFDataset,
    title_in_source="trpakov/chest-xray-classification",
    name="full",
)
dataset_provider.register_dataset(
    "sports-classification",
    HFDataset,
    title_in_source="HES-XPLAIN/SportsImageClassification",
)
dataset_provider.register_dataset("geo-landmarks", HFDataset, title_in_source="Qdrant/google-landmark-geo")

dataset_provider.register_dataset(
    "rsicd", EncordDataset, project_hash="46ba913e-1428-48ef-be7f-2553e69bc1e6", classification_hash="4f6cf0c8"
)
