"""Auditoría utilidad vs. riesgo de privacidad de datos sintéticos."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors


def fidelity(real: pd.DataFrame, synth: pd.DataFrame) -> dict:
    num = real.select_dtypes("number").columns
    ks = np.mean([ks_2samp(real[c], synth[c]).statistic for c in num])
    corr_err = np.abs(real[num].corr().values - synth[num].corr().values)
    return {"ks_mean": float(ks), "corr_mae": float(np.nanmean(corr_err))}


def utility_tstr(synth: pd.DataFrame, real_test: pd.DataFrame, label: str) -> dict:
    """Train-on-Synthetic, Test-on-Real."""
    Xs, ys = synth.drop(columns=[label]), synth[label]
    Xr, yr = real_test.drop(columns=[label]), real_test[label]
    Xs, Xr = pd.get_dummies(Xs), pd.get_dummies(Xr)
    Xs, Xr = Xs.align(Xr, join="left", axis=1, fill_value=0)
    pred = RandomForestClassifier(200, n_jobs=-1, random_state=0).fit(Xs, ys).predict(Xr)
    return {"f1": f1_score(yr, pred, average="macro"),
            "precision": precision_score(yr, pred, average="macro", zero_division=0),
            "recall": recall_score(yr, pred, average="macro", zero_division=0)}


def membership_inference_auc(X_train, X_holdout, X_synth) -> float:
    """Ataque por distancia al registro sintético más cercano. AUC≈0.5 = sin fuga; >0.6 sospechoso."""
    nn_ = NearestNeighbors(n_neighbors=1).fit(X_synth)
    d_in = nn_.kneighbors(X_train)[0].ravel()
    d_out = nn_.kneighbors(X_holdout)[0].ravel()
    y = np.r_[np.ones(len(d_in)), np.zeros(len(d_out))]
    return float(roc_auc_score(y, -np.r_[d_in, d_out]))


def duplicate_rate(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    merged = synth.merge(real.drop_duplicates(), how="inner")
    return len(merged) / max(len(synth), 1)


def full_report(real_train, real_holdout, synth, label=None) -> dict:
    rep = {"fidelity": fidelity(real_train, synth), "duplicates": duplicate_rate(real_train, synth)}
    if label:
        rep["utility"] = utility_tstr(synth, real_holdout, label)
    num = real_train.select_dtypes("number").columns
    rep["mia_auc"] = membership_inference_auc(real_train[num].values, real_holdout[num].values, synth[num].values)
    return rep
