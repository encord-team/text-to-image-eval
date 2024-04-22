import numpy as np

from clip_eval.common import EmbeddingDefinition, Embeddings, Split
from clip_eval.compute import compute_embeddings_from_definition

if __name__ == "__main__":
    def_ = EmbeddingDefinition(
        model="weird_this  with  / stuff \\whatever",
        dataset="hello there dataset",
    )
    def_.embedding_path(Split.TRAIN).parent.mkdir(exist_ok=True, parents=True)

    images = np.random.randn(100, 20).astype(np.float32)
    labels = np.random.randint(0, 10, size=(100,))
    classes = np.random.randn(10, 20).astype(np.float32)
    emb = Embeddings(images=images, labels=labels, classes=classes)
    # emb.to_file(def_.embedding_path(Split.TRAIN))
    def_.save_embeddings(emb, split=Split.TRAIN, overwrite=True)
    new_emb = def_.load_embeddings(Split.TRAIN)

    assert new_emb is not None
    assert np.allclose(new_emb.images, images)
    assert np.allclose(new_emb.labels, labels)

    from pydantic import ValidationError

    try:
        Embeddings(
            images=np.random.randn(100, 20).astype(np.float32),
            labels=np.random.randint(0, 10, size=(100,)),
            classes=np.random.randn(10, 30).astype(np.float32),
        )
        raise AssertionError()
    except ValidationError:
        pass

    def_ = EmbeddingDefinition(
        model="clip",
        dataset="LungCancer4Types",
    )
    embeddings = compute_embeddings_from_definition(def_, Split.VALIDATION)
    def_.save_embeddings(embeddings, split=Split.VALIDATION, overwrite=True)
