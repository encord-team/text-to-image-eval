import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from natsort import natsorted, ns

from clip_eval.constants import CACHE_PATH, SOURCES_PATH

from .base import Dataset, DatasetDefinitionSpec, Split
from .utils import load_class_from_path


class DatasetProvider:
    __instance = None
    __global_settings: dict[str, Any] = dict()
    __known_dataset_types: dict[tuple[Path, str], Any] = dict()

    def __init__(self):
        self._datasets = {}

    @classmethod
    def prepare(cls):
        if cls.__instance is None:
            cls.__instance = cls()
            cls.register_datasets_from_sources_dir(SOURCES_PATH.DATASET_INSTANCE_DEFINITIONS)
            # Global settings
            cls.__instance.add_global_setting("cache_dir", CACHE_PATH)
        return cls.__instance

    @classmethod
    def global_settings(cls) -> dict:
        return deepcopy(cls.__global_settings)

    @classmethod
    def add_global_setting(cls, name: str, value: Any) -> None:
        cls.__global_settings[name] = value

    @classmethod
    def remove_global_setting(cls, name: str) -> None:
        cls.__global_settings.pop(name, None)

    @classmethod
    def register_dataset(cls, source: type[Dataset], title: str, split: Split | None = None, **kwargs) -> None:
        instance = cls.prepare()
        if split is None:
            # One dataset with all the split definitions
            instance._datasets[title] = (source, kwargs)
        else:
            # One dataset is defined per split
            kwargs.update(split=split)
            instance._datasets[(title, split)] = (source, kwargs)

    @classmethod
    def register_dataset_from_json_definition(cls, json_definition: Path) -> None:
        spec = DatasetDefinitionSpec(**json.loads(json_definition.read_text(encoding="utf-8")))
        if not spec.module_path.is_absolute():  # Handle relative module paths
            spec.module_path = (json_definition.parent / spec.module_path).resolve()

        # Fetch the class of the dataset type stated in the definition
        dataset_type = cls.__known_dataset_types.get((spec.module_path, spec.dataset_type))
        if dataset_type is None:
            dataset_type = load_class_from_path(spec.module_path.as_posix(), spec.dataset_type)
            if not issubclass(dataset_type, Dataset):
                raise ValueError(
                    f"Dataset type specified in the JSON definition file `{json_definition.as_posix()}` "
                    f"does not inherit from the base class `Dataset`"
                )
            cls.__known_dataset_types[(spec.module_path, spec.dataset_type)] = dataset_type
        cls.register_dataset(dataset_type, **spec.model_dump(exclude={"module_path", "dataset_type"}))

    @classmethod
    def register_datasets_from_sources_dir(cls, source_dir: Path) -> None:
        for f in source_dir.glob("*.json"):
            cls.register_dataset_from_json_definition(f)

    @classmethod
    def get_dataset(cls, title: str, split: Split) -> Dataset:
        instance = cls.prepare()
        if (title, split) in instance._datasets:
            # The split corresponds to fetching a whole dataset (one-to-one relationship)
            dict_key = (title, split)
            split = Split.ALL  # Ensure to read the whole dataset
        elif title in instance._datasets:
            # The dataset internally knows how to determine the split
            dict_key = title
        else:
            raise ValueError(f"Unrecognized dataset: {title}")

        source, kwargs = instance._datasets[dict_key]
        # Apply global settings. Values of local settings take priority when local and global settings share keys.
        kwargs_with_global_settings = cls.global_settings() | kwargs
        return source(title, split=split, **kwargs_with_global_settings)

    @classmethod
    def list_dataset_titles(cls) -> list[str]:
        dataset_titles = [
            dict_key[0] if isinstance(dict_key, tuple) else dict_key for dict_key in cls.prepare()._datasets.keys()
        ]
        return natsorted(set(dataset_titles), alg=ns.IGNORECASE)
