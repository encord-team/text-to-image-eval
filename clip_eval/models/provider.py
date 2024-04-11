import json
from pathlib import Path
from typing import Any

from natsort import natsorted, ns

from clip_eval.constants import SOURCES_PATH
from clip_eval.dataset.utils import load_class_from_path

from .base import Model


class ModelProvider:
    def __init__(self) -> None:
        self._models = {}
        self.__known_model_types: dict[tuple[Path, str], Any] = dict()

    def register_model(self, title: str, source: type[Model], **kwargs):
        self._models[title] = (source, kwargs)

    def register_model_from_json_definition(self, json_definition: Path) -> None:
        model_params: dict = json.loads(json_definition.read_text(encoding="utf-8"))
        module_path_str: str | None = model_params.pop("module_path", None)
        model_type_name: str | None = model_params.pop("model_type", None)
        if module_path_str is None or model_type_name is None:
            raise ValueError(
                f"Missing required fields `module_path` or `model_type` in "
                f"the JSON definition file: {json_definition.as_posix()}"
            )

        # Handle relative module paths
        module_path = Path(module_path_str)
        if not module_path.is_absolute():
            module_path = (json_definition.parent / module_path).resolve()

        # Fetch the class of the model type stated in the definition
        model_type = self.__known_model_types.get((module_path, model_type_name))
        if model_type is None:
            model_type = load_class_from_path(module_path.as_posix(), model_type_name)
            if not issubclass(model_type, Model):
                raise ValueError(
                    f"Model type specified in the JSON definition file `{json_definition.as_posix()}` "
                    f"does not inherit from the base class `Model`"
                )
            self.__known_model_types[(module_path, model_type_name)] = model_type
        self.register_model(model_type, **model_params)

    def register_models_from_sources_dir(self, source_dir: Path) -> None:
        for f in source_dir.glob("*.json"):
            self.register_model_from_json_definition(f)

    def get_model(self, title: str) -> Model:
        if title not in self._models:
            raise ValueError(f"Unrecognized model: {title}")
        source, kwargs = self._models[title]
        return source(title, **kwargs)

    def list_model_titles(self) -> list[str]:
        return natsorted(self._models.keys(), alg=ns.IGNORECASE)


model_provider = ModelProvider()
model_provider.register_models_from_sources_dir(SOURCES_PATH.MODEL_INSTANCE_DEFINITIONS)
