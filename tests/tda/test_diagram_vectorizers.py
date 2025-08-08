import numpy as np
from src.tda.vectorizers.diagram_vectorizers import (
    PersistenceImageConfig, PersistenceImageVectorizer,
    PersistenceLandscapeConfig, PersistenceLandscapeVectorizer,
)


def test_persistence_image_vectorizer_shapes_and_stability(sample_persistence_diagram):
    cfg = PersistenceImageConfig(resolution=(12, 10), sigma=0.1)
    vec = PersistenceImageVectorizer(cfg)
    imgs1 = [vec.transform_diagram(d) for d in sample_persistence_diagram]
    imgs2 = [vec.transform_diagram(d) for d in sample_persistence_diagram]
    expected_len = 12 * 10
    for a, b in zip(imgs1, imgs2):
        assert a.shape[0] == expected_len
        assert np.allclose(a, b), "Persistence image not stable across repeated transforms"


def test_persistence_image_empty_diagram():
    cfg = PersistenceImageConfig(resolution=(8, 6))
    vec = PersistenceImageVectorizer(cfg)
    empty = np.empty((0, 2))
    img = vec.transform_diagram(empty)
    assert img.shape[0] == 8 * 6
    assert np.count_nonzero(img) == 0


def test_persistence_landscape_vectorizer(sample_persistence_diagram):
    cfg = PersistenceLandscapeConfig(resolution=50, k_layers=3)
    vec = PersistenceLandscapeVectorizer(cfg)
    # Use first diagram (H0) only
    arr = vec.transform_diagram(sample_persistence_diagram[0])
    assert arr.shape[0] == 50 * 3
    # Stability
    arr2 = vec.transform_diagram(sample_persistence_diagram[0])
    assert np.allclose(arr, arr2)


def test_persistence_landscape_empty():
    cfg = PersistenceLandscapeConfig(resolution=30, k_layers=2)
    vec = PersistenceLandscapeVectorizer(cfg)
    empty = np.empty((0, 2))
    arr = vec.transform_diagram(empty)
    assert arr.shape[0] == 30 * 2
    assert np.count_nonzero(arr) == 0
