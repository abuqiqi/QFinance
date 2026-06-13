import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC


def test_model_round_trip(tmp_path):
    x = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]] * 4, dtype=float)
    y = np.asarray([0, 0, 1, 1] * 4)
    model = CalibratedClassifierCV(SVC(), cv=2).fit(x, y)
    before = model.predict_proba(x)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(before, joblib.load(path).predict_proba(x))

