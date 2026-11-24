"""Auditoría de utilidad, fidelidad y riesgo de privacidad de datos sintéticos.

Las métricas de este módulo son diagnósticos empíricos; no sustituyen una prueba
formal de privacidad diferencial ni una evaluación de seguridad independiente.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _validate_frames(real: pd.DataFrame, synth: pd.DataFrame) -> None:
    if not isinstance(real, pd.DataFrame) or not isinstance(synth, pd.DataFrame):
        raise TypeError("real y synth deben ser pandas.DataFrame")
    if real.empty or synth.empty:
        raise ValueError("real y synth deben contener al menos una fila")
    missing = real.columns.difference(synth.columns)
    if len(missing):
        raise ValueError(f"Faltan columnas en synth: {missing.tolist()}")


def fidelity(real: pd.DataFrame, synth: pd.DataFrame) -> dict:
    """Compara variables numéricas (KS/correlación) y categóricas (TV).

    ``ks_mean`` y ``corr_mae`` son ``None`` cuando no hay variables numéricas
    evaluables. ``categorical_tv_mean`` es ``None`` cuando no hay categóricas.
    La distancia de variación total (TV) está entre 0 (idéntico) y 1.
    """
    _validate_frames(real, synth)
    numeric = [c for c in real.select_dtypes(include="number").columns if c in synth.columns
               and pd.api.types.is_numeric_dtype(synth[c])]
    categorical = [c for c in real.columns if c not in numeric]

    ks_values = []
    for column in numeric:
        left = pd.to_numeric(real[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        right = pd.to_numeric(synth[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(left) and len(right):
            ks_values.append(float(ks_2samp(left, right).statistic))

    if ks_values:
        corr_delta = (real[numeric].corr() - synth[numeric].corr()).abs().to_numpy()
        finite_corr = corr_delta[np.isfinite(corr_delta)]
        corr_mae = float(finite_corr.mean()) if finite_corr.size else None
        ks_mean = float(np.mean(ks_values))
    else:
        ks_mean, corr_mae = None, None

    tv_values = []
    for column in categorical:
        real_dist = real[column].value_counts(normalize=True, dropna=False)
        synth_dist = synth[column].value_counts(normalize=True, dropna=False)
        categories = real_dist.index.union(synth_dist.index)
        tv_values.append(float((real_dist.reindex(categories, fill_value=0)
                                - synth_dist.reindex(categories, fill_value=0)).abs().sum() / 2))

    return {
        "ks_mean": ks_mean,
        "corr_mae": corr_mae,
        "categorical_tv_mean": float(np.mean(tv_values)) if tv_values else None,
    }


def utility_tstr(synth: pd.DataFrame, real_test: pd.DataFrame, label: str) -> dict:
    """Train-on-Synthetic, Test-on-Real (TSTR), con esquema alineado."""
    if label not in synth.columns or label not in real_test.columns:
        raise ValueError(f"La etiqueta {label!r} debe existir en ambos conjuntos")
    if synth.empty or real_test.empty:
        raise ValueError("synth y real_test deben contener al menos una fila")
    Xs, ys = synth.drop(columns=[label]), synth[label]
    Xr, yr = real_test.drop(columns=[label]), real_test[label]
    Xs, Xr = pd.get_dummies(Xs), pd.get_dummies(Xr)
    Xs, Xr = Xs.align(Xr, join="outer", axis=1, fill_value=0)
    model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
    pred = model.fit(Xs, ys).predict(Xr)
    return {
        "f1": float(f1_score(yr, pred, average="macro", zero_division=0)),
        "precision": float(precision_score(yr, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(yr, pred, average="macro", zero_division=0)),
    }


def _feature_matrix(values, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} debe ser una matriz 2D no vacía")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contiene NaN o valores infinitos")
    return matrix


def membership_inference_auc(X_train, X_holdout, X_synth) -> float:
    """AUC de ataque heurístico basado en distancia al sintético más cercano.

    AUC cercano a 0.5 indica que este ataque concreto no distingue bien los
    conjuntos; no demuestra ausencia de fuga ni constituye una garantía formal.
    Las distancias se normalizan usando únicamente el conjunto de entrenamiento.
    """
    train = _feature_matrix(X_train, "X_train")
    holdout = _feature_matrix(X_holdout, "X_holdout")
    synthetic = _feature_matrix(X_synth, "X_synth")
    if train.shape[1] != holdout.shape[1] or train.shape[1] != synthetic.shape[1]:
        raise ValueError("X_train, X_holdout y X_synth deben tener igual número de columnas")

    scaler = StandardScaler().fit(train)
    train, holdout, synthetic = (scaler.transform(x) for x in (train, holdout, synthetic))
    nn_model = NearestNeighbors(n_neighbors=1).fit(synthetic)
    d_in = nn_model.kneighbors(train)[0].ravel()
    d_out = nn_model.kneighbors(holdout)[0].ravel()
    y_true = np.r_[np.ones(len(d_in)), np.zeros(len(d_out))]
    return float(roc_auc_score(y_true, -np.r_[d_in, d_out]))


def duplicate_rate(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """Proporción de filas sintéticas idénticas a una fila real."""
    _validate_frames(real, synth)
    columns = list(real.columns)
    if set(columns) != set(synth.columns):
        raise ValueError("real y synth deben tener las mismas columnas para comparar duplicados")
    real_rows = real[columns].drop_duplicates()
    synthetic_rows = synth[columns]
    duplicates = synthetic_rows.merge(real_rows, how="inner", on=columns)
    return float(len(duplicates) / len(synthetic_rows))


def full_report(real_train, real_holdout, synth, label=None) -> dict:
    """Genera métricas de fidelidad, utilidad opcional y riesgo empírico.

    La comprobación de membresía utiliza solo columnas numéricas y es una
    aproximación; si no existen, ``mia_auc`` queda como ``None``.
    """
    _validate_frames(real_train, synth)
    _validate_frames(real_holdout, synth)
    report = {
        "fidelity": fidelity(real_train, synth),
        "duplicates": duplicate_rate(real_train, synth),
    }
    if label is not None:
        report["utility"] = utility_tstr(synth, real_holdout, label)

    numeric = [c for c in real_train.select_dtypes(include="number").columns if c in synth.columns
               and c in real_holdout.columns and pd.api.types.is_numeric_dtype(synth[c])]
    if numeric:
        report["mia_auc"] = membership_inference_auc(
            real_train[numeric].to_numpy(), real_holdout[numeric].to_numpy(), synth[numeric].to_numpy()
        )
    else:
        report["mia_auc"] = None
    return report
