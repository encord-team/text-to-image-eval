import numpy as np
import torch
from transformers import CLIPModel as HF_ClipModel
from transformers import CLIPProcessor

options = {
    "clip": "openai/clip-vit-large-patch14-336",
    "pubmed": "flaviagiammarino/pubmed-clip-vit-base-patch32",
    "plip": "vinid/plip",
    "flax": "flax-community/clip-rsicd-v4",
    "street": "geolocal/StreetCLIP",
    # "biomed": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", # doesn't work - needs "open clop style loading" https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/blob/main/biomed_clip_example.ipynb
    "apple": "apple/DFN5B-CLIP-ViT-H-14",  #  this is huge
    # "pmc": "ryanyip7777/pmc_vit_l_14",  # pmc open access dataset also open-clip
}
short_name = "clip"
model_name = options[short_name]


class CLIPModel:
    def __init__(self, model_name: str, device: str = torch.device("cpu")) -> None:
        self.model = HF_ClipModel.from_pretrained(model_name).to(device)
        self.title = model_name
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.process_fn = self.define_process_fn()

    def define_process_fn(self):
        def process_fn(batch):
            images = [i.convert("RGB") for i in batch["image"]]
            batch["image"] = [
                self.processor(images=[i], return_tensors="pt")
                .to("cuda")
                .pixel_values.squeeze()
                for i in images
            ]
            return batch

        return process_fn

    @classmethod
    def load_model(cls, model_name: str):
        return CLIPModel(model_name=model_name)
