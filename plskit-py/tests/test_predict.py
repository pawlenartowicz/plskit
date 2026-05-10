import numpy as np
import plskit


def test_predict_round_trip():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5)); y = X[:, 0] + 0.1 * rng.normal(size=60)
    m = plskit.pls1_fit(X, y, k=2)
    y_hat = plskit.pls1_predict(m, X)
    # In-sample R² > 0.9 with this signal
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float(((y - y_hat) ** 2).sum())
    assert 1 - ss_res / ss_tot > 0.9
