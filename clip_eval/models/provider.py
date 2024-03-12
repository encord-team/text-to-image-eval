from .CLIP_model import CLIPModel, ClosedCLIPModel, OpenCLIPModel, SiglipModel
from .local import LocalCLIPModel


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
model_provider.register_model("clip", ClosedCLIPModel, title_in_source="openai/clip-vit-large-patch14-336")
model_provider.register_model("plip", ClosedCLIPModel, title_in_source="vinid/plip")
model_provider.register_model(
    "pubmed",
    ClosedCLIPModel,
    title_in_source="flaviagiammarino/pubmed-clip-vit-base-patch32",
)
model_provider.register_model(
    "tinyclip",
    ClosedCLIPModel,
    title_in_source="wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M",
)
model_provider.register_model("fashion", ClosedCLIPModel, title_in_source="patrickjohncyh/fashion-clip")
model_provider.register_model("rscid", ClosedCLIPModel, title_in_source="flax-community/clip-rsicd")
model_provider.register_model("street", ClosedCLIPModel, title_in_source="geolocal/StreetCLIP")

model_provider.register_model("apple", OpenCLIPModel, title_in_source="hf-hub:apple/DFN5B-CLIP-ViT-H-14")
model_provider.register_model("eva-clip", OpenCLIPModel, title_in_source="BAAI/EVA-CLIP-8B-448")
model_provider.register_model("bioclip", OpenCLIPModel, title_in_source="hf-hub:imageomics/bioclip")
model_provider.register_model("vit-b-32-laion2b", OpenCLIPModel, title_in_source="ViT-B-32", pretrained="laion2b_e16")

model_provider.register_model("siglip_small", SiglipModel, title_in_source="google/siglip-base-patch16-224")
model_provider.register_model("siglip_large", SiglipModel, title_in_source="google/siglip-large-patch16-256")

# Local sources
model_provider.register_model("rsicd-encord", LocalCLIPModel, title_in_source="ViT-B/32")
