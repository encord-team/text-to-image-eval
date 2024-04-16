from pathlib import Path
from typing import Any

from natsort import natsorted, ns

from clip_eval.constants import SOURCES_PATH
from clip_eval.dataset.utils import load_class_from_path

from .base import Model, ModelDefinitionSpec


class ModelProvider:
    __instance = None
    __known_model_types: dict[tuple[Path, str], Any] = dict()

    def __init__(self) -> None:
        self._models = {}

    @classmethod
    def prepare(cls):
        if cls.__instance is None:
            cls.__instance = cls()
            cls.register_models_from_sources_dir(SOURCES_PATH.MODEL_INSTANCE_DEFINITIONS)
        return cls.__instance

    @classmethod
    def register_model(cls, source: type[Model], title: str, **kwargs):
        cls.prepare()._models[title] = (source, kwargs)

    @classmethod
    def register_model_from_json_definition(cls, json_definition: Path) -> None:
        spec = ModelDefinitionSpec.model_validate_json(json_definition.read_text(encoding="utf-8"))
        if not spec.module_path.is_absolute():  # Handle relative module paths
            spec.module_path = (json_definition.parent / spec.module_path).resolve()
        if not spec.module_path.is_file():
            raise FileNotFoundError(
                f"Could not find the specified module path at `{spec.module_path.as_posix()}` "
                f"when registering model `{spec.title}`"
            )

        # Fetch the class of the model type stated in the definition
        model_type = cls.__known_model_types.get((spec.module_path, spec.model_type))
        if model_type is None:
            model_type = load_class_from_path(spec.module_path.as_posix(), spec.model_type)
            if not issubclass(model_type, Model):
                raise ValueError(
                    f"Model type specified in the JSON definition file `{json_definition.as_posix()}` "
                    f"does not inherit from the base class `Model`"
                )
            cls.__known_model_types[(spec.module_path, spec.model_type)] = model_type
        cls.register_model(model_type, **spec.model_dump(exclude={"module_path", "model_type"}))

    @classmethod
    def register_models_from_sources_dir(cls, source_dir: Path) -> None:
        for f in source_dir.glob("*.json"):
            cls.register_model_from_json_definition(f)

    @classmethod
    def get_model(cls, title: str) -> Model:
        instance = cls.prepare()
        if title not in instance._models:
            raise ValueError(f"Unrecognized model: {title}")
        source, kwargs = instance._models[title]
        return source(title, **kwargs)

    @classmethod
    def list_model_titles(cls) -> list[str]:
        return natsorted(cls.prepare()._models.keys(), alg=ns.IGNORECASE)
