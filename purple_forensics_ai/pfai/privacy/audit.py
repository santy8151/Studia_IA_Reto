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
    numeric = [
        c
        for c in real.select_dtypes(include="number").columns
        if c in synth.columns and pd.api.types.is_numeric_dtype(synth[c])
    ]
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
        tv_values.append(
            float(
                (
                    real_dist.reindex(categories, fill_value=0)
                    - synth_dist.reindex(categories, fill_value=0)
                )
                .abs()
                .sum()
                / 2
            )
        )

    return {
        "ks_mean": ks_mean,
        "corr_mae": corr_mae,
        "categorical_tv_mean": float(np.mean(tv_values)) if tv_values else None,
    }


def _train_eval_rf(train_df: pd.DataFrame, test_df: pd.DataFrame, label: str) -> dict:
    if label not in train_df.columns or label not in test_df.columns:
        raise ValueError(f"La etiqueta {label!r} debe existir en ambos conjuntos")
    if train_df.empty or test_df.empty:
        raise ValueError("train_df y test_df deben contener al menos una fila")

    Xs, ys = train_df.drop(columns=[label]), train_df[label]
    Xr, yr = test_df.drop(columns=[label]), test_df[label]
    Xs, Xr = pd.get_dummies(Xs), pd.get_dummies(Xr)
    Xs, Xr = Xs.align(Xr, join="outer", axis=1, fill_value=0)
    Xs = Xs.fillna(0)
    Xr = Xr.fillna(0)

    if ys.nunique(dropna=True) < 2:
        const_val = ys.iloc[0]
        pred = np.repeat(const_val, len(yr))
    else:
        model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
        pred = model.fit(Xs, ys).predict(Xr)

    return {
        "f1": float(f1_score(yr, pred, average="macro", zero_division=0)),
        "precision": float(precision_score(yr, pred, average="macro", zero_division=0)),
        "recall": float(recall_score(yr, pred, average="macro", zero_division=0)),
    }


def utility_tstr(synth: pd.DataFrame, real_test: pd.DataFrame, label: str) -> dict:
    """Train-on-Synthetic, Test-on-Real (TSTR), con esquema alineado."""
    return _train_eval_rf(synth, real_test, label)


def utility_trtr(real_train: pd.DataFrame, real_test: pd.DataFrame, label: str) -> dict:
    """Train-on-Real, Test-on-Real (TRTR), línea base superior de utilidad."""
    return _train_eval_rf(real_train, real_test, label)


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


def distance_to_closest_record(X_train, X_holdout, X_synth) -> dict:
    """Calcula DCR (Distance to Closest Record) y NNDR (Nearest Neighbor Distance Ratio).

    Valores de DCR muy cercanos a 0 o un DCR(train) mucho menor que DCR(holdout)
    sugieren riesgo de memorización respecto al conjunto de entrenamiento.
    """
    train = _feature_matrix(X_train, "X_train")
    holdout = _feature_matrix(X_holdout, "X_holdout")
    synthetic = _feature_matrix(X_synth, "X_synth")
    if train.shape[1] != holdout.shape[1] or train.shape[1] != synthetic.shape[1]:
        raise ValueError("X_train, X_holdout y X_synth deben tener igual número de columnas")

    scaler = StandardScaler().fit(train)
    train_s, holdout_s, synth_s = (scaler.transform(x) for x in (train, holdout, synthetic))

    k_tr = min(2, len(train_s))
    nn_tr = NearestNeighbors(n_neighbors=k_tr).fit(train_s)
    dist_tr, _ = nn_tr.kneighbors(synth_s)

    nn_ho = NearestNeighbors(n_neighbors=1).fit(holdout_s)
    dist_ho, _ = nn_ho.kneighbors(synth_s)

    d1_tr = dist_tr[:, 0]
    d1_ho = dist_ho[:, 0]
    nndr = float(np.mean(d1_tr / np.maximum(dist_tr[:, 1], 1e-8))) if k_tr == 2 else None
    ho_mean = float(np.mean(d1_ho))
    return {
        "dcr_train_mean": float(np.mean(d1_tr)),
        "dcr_train_p05": float(np.quantile(d1_tr, 0.05)),
        "dcr_holdout_mean": ho_mean,
        "dcr_ratio": float(np.mean(d1_tr) / max(ho_mean, 1e-8)),
        "nndr_train_mean": nndr,
    }


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


def full_report(real_train: pd.DataFrame, real_holdout: pd.DataFrame, synth: pd.DataFrame, label: str | None = None) -> dict:
    """Genera métricas de fidelidad, utilidad opcional (TSTR/TRTR) y riesgo empírico.

    La comprobación de membresía y DCR utilizan solo columnas numéricas sin NaN;
    si no existen columnas numéricas evaluables, ``mia_auc`` y ``dcr`` quedan como ``None``.
    """
    _validate_frames(real_train, synth)
    _validate_frames(real_holdout, synth)
    report: dict = {
        "fidelity": fidelity(real_train, synth),
        "duplicates": duplicate_rate(real_train, synth),
    }
    if label is not None:
        tstr = utility_tstr(synth, real_holdout, label)
        trtr = utility_trtr(real_train, real_holdout, label)
        report["utility"] = tstr
        report["utility_baseline"] = trtr
        report["utility_retention"] = (
            float(tstr["f1"] / trtr["f1"]) if trtr["f1"] > 0 else None
        )

    numeric = [
        c
        for c in real_train.select_dtypes(include="number").columns
        if c in synth.columns
        and c in real_holdout.columns
        and pd.api.types.is_numeric_dtype(synth[c])
        and pd.api.types.is_numeric_dtype(real_holdout[c])
    ]
    if numeric:
        tr_num = real_train[numeric].dropna().to_numpy()
        ho_num = real_holdout[numeric].dropna().to_numpy()
        sy_num = synth[numeric].dropna().to_numpy()
        if len(tr_num) and len(ho_num) and len(sy_num):
            report["mia_auc"] = membership_inference_auc(tr_num, ho_num, sy_num)
            report["dcr"] = distance_to_closest_record(tr_num, ho_num, sy_num)
        else:
            report["mia_auc"] = None
            report["dcr"] = None
    else:
        report["mia_auc"] = None
        report["dcr"] = None
    return report


def privacy_utility_verdict(report: dict, privacy_budget: dict | None = None) -> dict:
    """Resume de forma interpretable el balance utilidad vs. riesgo de privacidad."""
    notes: list[str] = []
    fid = report.get("fidelity", {})
    ks = fid.get("ks_mean")
    dups = report.get("duplicates", 0.0)
    mia = report.get("mia_auc")
    retention = report.get("utility_retention")

    if ks is not None and ks > 0.25:
        notes.append(f"KS medio alto ({ks:.3f}): fidelidad marginal mejorable.")
    if dups > 0.0:
        notes.append(f"Se detectaron {dups:.1%} filas sintéticas idénticas al entrenamiento.")
    if mia is not None and mia > 0.65:
        notes.append(f"MIA AUC={mia:.3f} (>0.65): riesgo empírico de inferencia de membresía.")
    if retention is not None and retention < 0.75:
        notes.append(f"Retención de utilidad TSTR/TRTR baja ({retention:.1%}).")

    formal_dp = bool(privacy_budget and privacy_budget.get("epsilon_spent") is not None)
    if formal_dp:
        notes.append(
            f"Garantía formal DP-SGD registrada: ε={privacy_budget['epsilon_spent']:.2f}, δ={privacy_budget['delta']}."
        )
    else:
        notes.append("Sin garantía formal (ε, δ): las métricas empíricas no garantizan privacidad diferencial.")

    return {
        "formal_dp": formal_dp,
        "empirical_leakage_risk": (
            "high" if (dups > 0.05 or (mia is not None and mia > 0.65))
            else "moderate" if (dups > 0.0 or (mia is not None and mia > 0.58))
            else "low"
        ),
        "notes": notes,
    }
