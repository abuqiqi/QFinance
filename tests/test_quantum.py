import numpy as np
import pytest


@pytest.mark.quantum
def test_quantum_kernel_smoke():
    pytest.importorskip("qiskit_machine_learning")
    from quantum_risk_classifier.quantum import build_quantum_kernel

    x = np.asarray([[0.1] * 8, [0.2] * 8])
    matrix = build_quantum_kernel(reps=1).evaluate(x_vec=x)
    assert matrix.shape == (2, 2)
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-8)
    np.testing.assert_allclose(np.diag(matrix), np.ones(2), atol=1e-6)


def test_kernel_statistics_and_threshold():
    from quantum_risk_classifier.quantum import _best_threshold, kernel_statistics

    matrix = np.asarray([[1.0, 0.2], [0.2, 1.0]])
    stats = kernel_statistics(matrix)
    assert stats["offdiag_mean"] == pytest.approx(0.2)
    threshold, score = _best_threshold(
        np.asarray([0, 0, 1, 1]), np.asarray([-2.0, -1.0, 0.5, 1.0])
    )
    assert score == 1.0
    assert -1.0 < threshold <= 0.5


def test_training_and_test_ids_must_be_disjoint():
    import pandas as pd
    from quantum_risk_classifier.quantum import assert_disjoint_sample_ids

    assert_disjoint_sample_ids(
        pd.DataFrame({"sample_id": ["train-1"]}),
        pd.DataFrame({"sample_id": ["test-1"]}),
    )
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_sample_ids(
            pd.DataFrame({"sample_id": ["same"]}),
            pd.DataFrame({"sample_id": ["same"]}),
        )
