from clip_eval.common import EmbeddingDefinition, Embeddings, Split
from clip_eval.dataset import Dataset, DatasetProvider
from clip_eval.model import Model, ModelProvider
from torch.utils.data import DataLoader


def compute_embeddings(model: Model, dataset: Dataset, batch_size: int = 50) -> Embeddings:
    dataset.set_transform(model.get_transform())
    dataloader = DataLoader(dataset, collate_fn=model.get_collate_fn(), batch_size=batch_size)

    image_embeddings, class_embeddings, labels = model.build_embedding(dataloader)
    embeddings = Embeddings(images=image_embeddings, classes=class_embeddings, labels=labels)
    return embeddings


def compute_embeddings_from_definition(definition: EmbeddingDefinition, split: Split) -> Embeddings:
    model = ModelProvider.get_model(definition.model)
    dataset = DatasetProvider.get_dataset(definition.dataset, split)
    return compute_embeddings(model, dataset)
