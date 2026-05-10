import numpy as np
import pytest
import plskit


def test_pls_kit_error_has_code_attr():
    X = np.zeros((10, 5)); y = np.zeros(9)
    with pytest.raises(plskit.PlsKitError) as ei:
        plskit.pls1_fit(X, y, k=2)
    assert hasattr(ei.value, "code")
    assert ei.value.code == "dimension_mismatch"
