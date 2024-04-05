import numpy as np

from clip_eval.common import ClassArray, Embeddings, ProbabilityArray
from clip_eval.evaluation.knn import WeightedKNNClassifier
from clip_eval.evaluation.utils import normalize, softmax


def slow_knn_predict(
    train_embeddings: Embeddings,
    val_embeddings: Embeddings,
    num_classes: int,
    k: int,
) -> tuple[ProbabilityArray, ClassArray]:
    train_image_embeddings = normalize(train_embeddings.images)
    val_image_embeddings = normalize(val_embeddings.images)
    n = val_image_embeddings.shape[0]

    # Retrieve the classes and distances of the `k` nearest training embeddings to each validation embedding
    all_dists = np.linalg.norm(val_image_embeddings[:, np.newaxis] - train_image_embeddings[np.newaxis, :], axis=-1)
    nearest_indices = np.argsort(all_dists, axis=1)[:, :k]
    dists = all_dists[np.arange(n)[:, np.newaxis], nearest_indices]
    nearest_classes = np.take(train_embeddings.labels, nearest_indices)

    # Calculate class votes from the distances (avoiding division by zero)
    max_value = np.finfo(np.float32).max
    scores = np.divide(1, np.square(dists), out=np.full_like(dists, max_value), where=dists != 0)
    weighted_count = np.zeros((n, num_classes), dtype=np.float32)
    for cls in range(num_classes):
        mask = nearest_classes == cls
        weighted_count[:, cls] = np.ma.masked_array(scores, mask=~mask).sum(axis=1)
    probabilities = softmax(weighted_count)
    return probabilities, np.argmax(probabilities, axis=1)


def test_weighted_knn_classifier():
    np.random.seed(42)

    train_embeddings = Embeddings(
        images=np.random.randn(100, 10).astype(np.float32),
        labels=np.random.randint(0, 10, size=(100,)),
    )
    val_embeddings = Embeddings(
        images=np.random.randn(20, 10).astype(np.float32),
        labels=np.random.randint(0, 10, size=(20,)),
    )
    knn = WeightedKNNClassifier(
        train_embeddings,
        val_embeddings,
        num_classes=10,
    )
    probs, pred_classes = knn.predict()

    test_probs, test_pred_classes = slow_knn_predict(
        train_embeddings,
        val_embeddings,
        num_classes=knn.num_classes,
        k=knn.k,
    )

    assert (pred_classes == test_pred_classes).all()
    assert np.isclose(probs, test_probs).all()
