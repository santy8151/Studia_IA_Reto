"""Codificación tabular mixta (numéricas + categóricas) <-> matriz float32.

Soporta imputación de nulos, transformación cuantil normalizada para variables
numéricas, one-hot encoding con categorías desconocidas para categóricas,
reconstrucción inversa (`inverse_transform`) para auditar muestras generadas
por modelos en espacio latente/DP, y serialización segura.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer


class TabularEncoder:
    """Codificador bidireccional para tablas mixtas de eventos y logs."""

    def __init__(
        self,
        numeric: Iterable[str] = (),
        categorical: Iterable[str] = (),
        max_quantiles: int = 200,
        random_state: int = 0,
    ):
        self.numeric = list(numeric)
        self.categorical = list(categorical)
        if not self.numeric and not self.categorical:
            raise ValueError("Debe especificarse al menos una columna numérica o categórica")
        self.max_quantiles = max_quantiles
        self.random_state = random_state
        self.ct: ColumnTransformer | None = None
        self.integer_cols_: set[str] = set()
        self.num_bounds_: dict[str, tuple[float, float]] = {}
        self.cat_slices_: list[tuple[str, int, int, np.ndarray]] = []
        self.n_num_: int = 0
        self.n_features_: int = 0

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        exclude: Iterable[str] = (),
        max_quantiles: int = 200,
        random_state: int = 0,
    ) -> "TabularEncoder":
        """Infiera columnas numéricas y categóricas desde un DataFrame."""
        excluded = set(exclude)
        cols = [c for c in df.columns if c not in excluded]
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        categorical = [c for c in cols if c not in numeric]
        return cls(numeric=numeric, categorical=categorical, max_quantiles=max_quantiles, random_state=random_state)

    def _require_fitted(self) -> None:
        if self.ct is None:
            raise RuntimeError("TabularEncoder no está entrenado; llama a fit() primero")

    def _validate_columns(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Se esperaba un pandas.DataFrame")
        if df.empty:
            raise ValueError("El DataFrame está vacío")
        expected = set(self.numeric) | set(self.categorical)
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")

    def fit(self, df: pd.DataFrame) -> "TabularEncoder":
        self._validate_columns(df)
        parts = []
        self.integer_cols_ = {
            c for c in self.numeric if pd.api.types.is_integer_dtype(df[c].dtype)
        }
        self.num_bounds_ = {}
        for c in self.numeric:
            series = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(series):
                self.num_bounds_[c] = (float(series.min()), float(series.max()))
            else:
                self.num_bounds_[c] = (0.0, 0.0)

        if self.numeric:
            nq = max(2, min(self.max_quantiles, len(df)))
            num_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "scaler",
                        QuantileTransformer(
                            output_distribution="normal",
                            n_quantiles=nq,
                            random_state=self.random_state,
                        ),
                    ),
                ]
            )
            parts.append(("num", num_pipe, self.numeric))

        if self.categorical:
            cat_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )
            parts.append(("cat", cat_pipe, self.categorical))

        work = df.copy()
        for c in self.numeric:
            work[c] = pd.to_numeric(work[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        for c in self.categorical:
            work[c] = work[c].astype(object)

        self.ct = ColumnTransformer(parts, sparse_threshold=0.0).fit(work)
        self.n_num_ = len(self.numeric)

        self.cat_slices_ = []
        offset = self.n_num_
        if self.categorical:
            ohe: OneHotEncoder = self.ct.named_transformers_["cat"].named_steps["ohe"]
            for col, cats in zip(self.categorical, ohe.categories_):
                width = len(cats)
                self.cat_slices_.append((col, offset, offset + width, np.asarray(cats)))
                offset += width
        self.n_features_ = offset
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        self._require_fitted()
        self._validate_columns(df)
        work = df.copy()
        for c in self.numeric:
            work[c] = pd.to_numeric(work[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        for c in self.categorical:
            work[c] = work[c].astype(object)
        return np.asarray(self.ct.transform(work), dtype=np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def inverse_transform(self, X: np.ndarray, clip_bounds: bool = True) -> pd.DataFrame:
        """Reconstruye un DataFrame tabular desde la matriz codificada ``float32``."""
        self._require_fitted()
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.n_features_:
            raise ValueError(f"Se esperaba matriz de forma (n, {self.n_features_}), recibido {arr.shape}")

        data: dict[str, np.ndarray] = {}
        if self.numeric:
            qt: QuantileTransformer = self.ct.named_transformers_["num"].named_steps["scaler"]
            num_raw = qt.inverse_transform(arr[:, : self.n_num_])
            for idx, col in enumerate(self.numeric):
                vals = num_raw[:, idx]
                if clip_bounds and col in self.num_bounds_:
                    lo, hi = self.num_bounds_[col]
                    vals = np.clip(vals, lo, hi)
                if col in self.integer_cols_:
                    vals = np.rint(vals).astype(np.int64)
                data[col] = vals

        for col, start, end, cats in self.cat_slices_:
            best_idx = np.argmax(arr[:, start:end], axis=1)
            data[col] = cats[best_idx]

        ordered_cols = [c for c in self.numeric + self.categorical if c in data]
        return pd.DataFrame(data, columns=ordered_cols)

    def get_feature_names_out(self) -> list[str]:
        self._require_fitted()
        return [str(x) for x in self.ct.get_feature_names_out()]

    def save(self, path: str | Path) -> Path:
        self._require_fitted()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "TabularEncoder":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"El archivo {path} no contiene un TabularEncoder válido")
        return obj
