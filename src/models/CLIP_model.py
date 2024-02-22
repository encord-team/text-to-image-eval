from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel as HF_ClipModel
from transformers import CLIPProcessor as HF_ClipProcessor

# from src.common import ClassArray, EmbeddingArray

OPTIONS = {
    "clip": "openai/clip-vit-large-patch14-336",
    "pubmed": "flaviagiammarino/pubmed-clip-vit-base-patch32",
    "plip": "vinid/plip",
    "flax": "flax-community/clip-rsicd-v4",
    "street": "geolocal/StreetCLIP",
    # doesn't work - needs "open clop style loading" https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/blob/main/biomed_clip_example.ipynb
    # "biomed": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    # "sigLIP": "timm/ViT-SO400M-14-SigLIP-384" doesn't worj needs more open_clip style loading
    "apple": "apple/DFN5B-CLIP-ViT-H-14",  #  this is huge
    # "pmc": "ryanyip7777/pmc_vit_l_14",  # pmc open access dataset also open-clip
    "eva-clip": "BAAI/EVA-CLIP-8B-448",  # This is ginormous
    "fashion": "patrickjohncyh/fashion-clip",
    "rscid": "flax-community/clip-rsicd",
    "bioclip": "imageomics/bioclip",
    "tinyclip": "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M",
}


class CLIPModel(ABC):
    def __init__(self, title: str, title_in_source: str | None = None, device: str | None = None, **kwargs) -> None:
        self.__title = title
        self.__title_in_source = title if title_in_source is None else title_in_source
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._check_device(device)
        self.__device = torch.device(device)

    @property
    def title(self) -> str:
        return self.__title

    @property
    def title_in_source(self) -> str:
        return self.__title_in_source

    @property
    def device(self) -> torch.device:
        return self.__device

    @abstractmethod
    def _setup(self, **kwargs):
        pass

    @abstractmethod
    # Would like to have this return tuple[ImageEmbeddings, ClassArray]
    def build_embedding(self, dataloader: DataLoader):
        pass

    @staticmethod
    def _check_device(device: str):
        # Check if the input device exists and is available
        if device not in {"cuda", "cpu"}:
            raise ValueError(f"Unrecognized device: {device}")
        if not getattr(torch, device).is_available():
            raise ValueError(f"Unavailable device: {device}")


class closed_CLIPModel(CLIPModel):
    def __init__(self, title: str, title_in_source: str, device: str | None = None) -> None:
        super().__init__(title, title_in_source, device)
        self._setup()

    def define_process_fn(self):
        def process_fn(batch):
            images = [i.convert("RGB") for i in batch["image"]]
            batch["image"] = [
                self.processor(images=[i], return_tensors="pt").to(self.device).pixel_values.squeeze() for i in images
            ]
            return batch

        return process_fn

    def _setup(self):
        self.model = HF_ClipModel.from_pretrained(self.title_in_source).to(self.device)
        self.processor = HF_ClipProcessor.from_pretrained(self.title_in_source)
        self.process_fn = self.define_process_fn()

    def build_embedding(self, dataloader: DataLoader):
        tmp_embeddings = []
        tmp_labels = []
        with torch.inference_mode():
            for batch in tqdm(dataloader, desc=f"Embedding dataset with {self.title}"):
                tmp_labels.append(batch["labels"])
                features = self.model.get_image_features(pixel_values=batch["pixel_values"])
                emb = (features / features.norm(p=2, dim=-1, keepdim=True)).squeeze()
                tmp_embeddings.append(emb.to("cpu"))
        image_embeddings = np.concatenate(tmp_embeddings, 0)
        tmp_labels = torch.concatenate(tmp_labels)
        labels = tmp_labels.numpy()
        return image_embeddings, labels


class open_CLIPModel(CLIPModel):
    def __init__(self, title: str, title_in_source: str | None = None, device: str | None = None, **kwargs) -> None:
        super().__init__(title, title_in_source, device, **kwargs)
        self._setup()

    def _setup(self, **kwargs):
        raise NotImplementedError("open Clip not implemented")

    def build_embedding(self, dataloader: DataLoader):
        raise NotImplementedError("open Clip not implemented")
