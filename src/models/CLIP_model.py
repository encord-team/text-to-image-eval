from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPModel as HF_ClipModel
from transformers import CLIPProcessor

from src.datasets import HFDataset
from src.types import ClassArray, EmbeddingArray, Embeddings

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
    def __init__(
        self, model_name: str, device: torch.Device = torch.device("cuda")
    ) -> None:
        self.model = HF_ClipModel.from_pretrained(model_name).to(device)
        self.title = model_name

    def embed(self, dataset: HFDataset, batch_size: int = 50) -> Embeddings:
        def _collate_fn(examples) -> Dict[str, torch.tensor]:
            images = []
            labels = []
            for example in examples:
                images.append(example["image"])
                labels.append(examples["label"])

            pixel_values = torch.stack(images)
            labels = torch.tensor(labels)
            return {"pixel_values": pixel_values, "labels": labels}

        dataloader = DataLoader(dataset, collate_fn=_collate_fn, batch_size=batch_size)
        tmp_embeddings = []
        tmp_labels = []
        with torch.inference_mode():
            for batch in tqdm(dataloader, desc=f"Embedding dataset with {self.title}"):
                tmp_labels.append(batch["label"])
                features = self.model.get_image_features(
                    pixel_values=batch["pixel_values"]
                )
                emb = (features / features.norm(p=2, dim=-1, keepdim=True)).squeeze()
                tmp_embeddings.append(emb)
        embeddings: EmbeddingArray = np.concatenate(tmp_embeddings, 0)
        labels: ClassArray = np.array(tmp_labels)
        embeddings = Embeddings(images=embeddings, labels=labels)
        return embeddings

    @classmethod
    def load_model(cls, model_name: str):
        return CLIPModel(model_name=model_name)
