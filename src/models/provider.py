from src.models.CLIP_model import CLIPModel, closed_CLIPModel, open_CLIPModel


class ModelProvider:
    def __init__(self) -> None:
        self._models = {}

    def register_model(self, title: str, source: type[CLIPModel], **kwargs):
        self._models[title] = (source, kwargs)

    def get_model(self, title: str) -> CLIPModel:
        if title not in self._models:
            raise ValueError(f"Unrecognized model: {title}")
        source, kwargs = self._models[title]
        return source(title, **kwargs)

    def list_model_names(self) -> list[str]:
        return list(self._models.keys())


model_provider = ModelProvider()
model_provider.register_model("clip", closed_CLIPModel, title_in_source="openai/clip-vit-large-patch14-336")
model_provider.register_model("plip", closed_CLIPModel, title_in_source="vinid/plip")
model_provider.register_model(
    "pubmed", closed_CLIPModel, title_in_source="flaviagiammarino/pubmed-clip-vit-base-patch32"
)
model_provider.register_model("bioclip", closed_CLIPModel, title_in_source="imageomics/bioclip")
model_provider.register_model(
    "tinyclip", closed_CLIPModel, title_in_source="wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M"
)
model_provider.register_model("fashion", closed_CLIPModel, title_in_source="patrickjohncyh/fashion-clip")
model_provider.register_model("rscid", closed_CLIPModel, title_in_source="flax-community/clip-rsicd")
model_provider.register_model("street", closed_CLIPModel, title_in_source="geolocal/StreetCLIP")
model_provider.register_model("apple", open_CLIPModel, title_in_source="apple/DFN5B-CLIP-ViT-H-14")
model_provider.register_model("eva-clip", open_CLIPModel, title_in_source="BAAI/EVA-CLIP-8B-448")
