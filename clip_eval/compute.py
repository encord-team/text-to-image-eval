from clip_eval.common import EmbeddingDefinition, Embeddings
from clip_eval.dataset import DatasetProvider, Split
from clip_eval.model import ModelProvider


def compute_embeddings_from_definition(definition: EmbeddingDefinition, split: Split) -> Embeddings:
    model = ModelProvider.get_model(definition.model)
    dataset = DatasetProvider.get_dataset(definition.dataset, split)
    return Embeddings.build_embedding(model, dataset)
