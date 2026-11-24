import numpy as np
import pandas as pd
import pytest

from pfai.privacy.audit import duplicate_rate, fidelity, full_report, membership_inference_auc, utility_tstr


def test_fidelity_reports_numeric_and_categorical_metrics():
    real = pd.DataFrame({"count": [1, 2, 3], "kind": ["a", "a", "b"]})
    synth = pd.DataFrame({"count": [1, 2, 4], "kind": ["a", "b", "b"]})
    report = fidelity(real, synth)
    assert 0 <= report["ks_mean"] <= 1
    assert report["corr_mae"] == 0.0
    assert report["categorical_tv_mean"] == pytest.approx(1 / 6)


def test_fidelity_handles_categorical_only_frames():
    report = fidelity(pd.DataFrame({"kind": ["a"]}), pd.DataFrame({"kind": ["b"]}))
    assert report["ks_mean"] is None
    assert report["corr_mae"] is None
    assert report["categorical_tv_mean"] == 1.0


def test_fidelity_rejects_missing_synthetic_columns():
    with pytest.raises(ValueError, match="Faltan columnas"):
        fidelity(pd.DataFrame({"x": [1]}), pd.DataFrame({"y": [1]}))


def test_membership_audit_validates_shape_and_returns_auc():
    train = np.array([[0.0, 0.0], [1.0, 1.0]])
    holdout = np.array([[10.0, 10.0], [11.0, 11.0]])
    synth = np.array([[0.1, 0.1], [1.1, 1.1]])
    assert 0 <= membership_inference_auc(train, holdout, synth) <= 1
    with pytest.raises(ValueError, match="igual número"):
        membership_inference_auc(train, holdout, np.ones((2, 1)))


def test_duplicate_rate_and_full_report_are_json_friendly():
    real = pd.DataFrame({"count": [1, 2], "label": ["ok", "bad"]})
    holdout = pd.DataFrame({"count": [3, 4], "label": ["ok", "bad"]})
    synth = pd.DataFrame({"count": [1, 5], "label": ["ok", "ok"]})
    assert duplicate_rate(real, synth) == 0.5
    report = full_report(real, holdout, synth, label="label")
    assert report["duplicates"] == 0.5
    assert set(report["utility"]) == {"f1", "precision", "recall"}
    assert np.isfinite(report["mia_auc"])


def test_tstr_rejects_missing_label():
    with pytest.raises(ValueError, match="etiqueta"):
        utility_tstr(pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [2]}), "label")
