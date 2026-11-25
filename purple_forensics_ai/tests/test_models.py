import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from pfai.models.lvgan import LVGAN, LVGANConfig
from pfai.models.moe import FAMILIES, MoEClassifier


def _toy_data(seed=0):
    rng = np.random.default_rng(seed)
    benign = rng.normal(0.0, 1.0, (400, 8)).astype(np.float32)
    dos = rng.normal(0.0, 1.0, (80, 8)).astype(np.float32)
    dos[:, 0] += 4.5
    probe = rng.normal(0.0, 1.0, (80, 8)).astype(np.float32)
    probe[:, 1] += 4.5
    probe[:, 2] -= 3.5
    return benign, dos, probe


def test_lvgan_fit_score_evasion_and_serialization(tmp_path):
    benign, dos, _ = _toy_data()
    X_tr, X_val = benign[:300], benign[300:]

    lv = LVGAN(LVGANConfig(d_in=8, d_z=4, hidden=32, epochs=20, batch_size=64))
    with pytest.raises(RuntimeError, match="Umbral no calibrado"):
        lv.predict(X_val)

    lv.fit(X_tr, verbose=False)
    thr = lv.calibrate(X_val, q=0.95)
    assert np.isfinite(thr)

    comp = lv.score_components(dos)
    assert set(comp) == {"total", "rec_error", "feat_error"}
    assert np.allclose(comp["total"], comp["rec_error"] + lv.cfg.lam_feat * comp["feat_error"])

    auc = roc_auc_score(np.r_[np.zeros(len(X_val)), np.ones(len(dos))], lv.score(np.vstack([X_val, dos])))
    assert auc > 0.90

    prior_samples = lv.sample(10)
    assert prior_samples.shape == (10, 8)
    assert np.isfinite(prior_samples).all()

    evasion = lv.generate_evasion_variants(dos[:10], steps=20, step_size=0.1)
    assert evasion.shape == (10, 8)
    assert np.mean(lv.score(evasion)) <= np.mean(lv.score(dos[:10]))

    ckpt = lv.save(tmp_path / "lvgan.pt")
    loaded = LVGAN.load(ckpt)
    assert loaded.threshold == pytest.approx(lv.threshold)
    assert np.allclose(loaded.score(dos[:5]), lv.score(dos[:5]))


def test_moe_fit_balance_utilization_and_serialization(tmp_path):
    benign, dos, probe = _toy_data(1)
    X = np.vstack([benign[:120], dos, probe])
    y = np.r_[np.zeros(120, dtype=int), np.ones(len(dos), dtype=int), np.full(len(probe), 2, dtype=int)]

    with pytest.raises(ValueError, match="top_k"):
        MoEClassifier(8, n_experts=2, top_k=3)

    moe = MoEClassifier(d_in=8, n_classes=len(FAMILIES), n_experts=4, top_k=2, hidden=32)
    moe.fit(X, y, epochs=30, lr=3e-3, class_weights="balanced", verbose=False)

    probs = moe.predict_proba(X)
    assert probs.shape == (len(X), len(FAMILIES))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    ev = moe.evaluate(X, y)
    assert ev["accuracy"] > 0.85
    assert ev["macro_f1"] > 0.80

    util = moe.expert_utilization(X, y)
    assert len(util["overall_load"]) == 4
    assert sum(util["overall_load"]) == pytest.approx(1.0)
    assert set(util["by_family"]) == {"benign", "dos", "probe"}

    ckpt = moe.save(tmp_path / "moe.pt")
    loaded = MoEClassifier.load(ckpt)
    assert np.allclose(loaded.predict_proba(X[:5]), probs[:5])
