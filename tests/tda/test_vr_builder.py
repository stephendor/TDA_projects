import numpy as np
from src.tda.builders.vr import VRPersistenceBuilder

def test_vr_builder_basic():
    pc = np.random.rand(100, 3)
    builder = VRPersistenceBuilder(maxdim=2)
    result = builder.compute(pc)
    assert "diagrams" in result
    assert len(result["diagrams"]) >= 1
    assert result["features"].ndim == 1
