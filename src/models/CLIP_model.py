import torch
from transformers import CLIPModel as HF_ClipModel
from transformers import CLIPProcessor

OPTIONS = {
    "clip": "openai/clip-vit-large-patch14-336",
    "pubmed": "flaviagiammarino/pubmed-clip-vit-base-patch32",
    "plip": "vinid/plip",
    "flax": "flax-community/clip-rsicd-v4",
    "street": "geolocal/StreetCLIP",
    # doesn't work - needs "open clop style loading" https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/blob/main/biomed_clip_example.ipynb
    # "biomed": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "apple": "apple/DFN5B-CLIP-ViT-H-14",  #  this is huge
    # "pmc": "ryanyip7777/pmc_vit_l_14",  # pmc open access dataset also open-clip
}


class CLIPModel:
    def __init__(self, model_title: str, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._check_device(device)

        self.device = torch.device(device)
        self.title = model_title
        if model_title not in OPTIONS:
            raise ValueError("Unsupported model name")
        model_title = OPTIONS.get(model_title, None)
        self.model = HF_ClipModel.from_pretrained(model_title).to(self.device)

        self.processor = CLIPProcessor.from_pretrained(model_title)
        self.process_fn = self.define_process_fn()

    @staticmethod
    def _check_device(device: str):
        # Check if the input device exists and is available
        if device not in {"cuda", "cpu"}:
            raise ValueError(f"Unrecognized device: {device}")
        if not getattr(torch, device).is_available():
            raise ValueError(f"Unavailable device: {device}")

    def define_process_fn(self):
        def process_fn(batch):
            images = [i.convert("RGB") for i in batch["image"]]
            batch["image"] = [
                self.processor(images=[i], return_tensors="pt").to(self.device).pixel_values.squeeze() for i in images
            ]
            return batch

        return process_fn
