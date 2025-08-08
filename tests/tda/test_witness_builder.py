import numpy as np
from src.tda.builders.witness import WitnessPersistenceBuilder

def test_witness_builder_basic():
    pc = np.random.rand(500, 4)
    builder = WitnessPersistenceBuilder(landmarks=64, maxdim=2)
    result = builder.compute(pc)
    assert "diagrams" in result
    assert result["landmark_count"] <= 64
    assert result["features"].ndim == 1
