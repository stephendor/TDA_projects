import numpy as np
import pandas as pd
from src.tda.vectorizers.window_tda_vectorizer import TDAWindowVectorizer, TDAWindowVectorizerConfig

def test_window_tda_vectorizer_basic():
    n = 300
    df = pd.DataFrame({
        'timestamp': np.linspace(0, 10, n),
        'FlowDuration': np.random.rand(n),
        'TotFwdPkts': np.random.randint(0,100,n),
        'TotBwdPkts': np.random.randint(0,100,n)
    })
    cfg = TDAWindowVectorizerConfig(feature_columns=['FlowDuration','TotFwdPkts','TotBwdPkts'], max_vr_points=150)
    vec = TDAWindowVectorizer(cfg)
    out = vec.vectorize(df)
    assert 'features' in out and out['features'].ndim == 1
    assert 'diagrams' in out
