"""Codificación tabular mixta (numéricas + categóricas) -> matriz float32."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer


class TabularEncoder:
    def __init__(self, numeric, categorical):
        self.numeric, self.categorical = numeric, categorical
        self.ct = None

    def fit(self, df: pd.DataFrame):
        parts = []
        if self.numeric:
            nq = max(10, min(200, len(df)))
            parts.append(("num", QuantileTransformer(output_distribution="normal", n_quantiles=nq), self.numeric))
        if self.categorical:
            parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical))
        self.ct = ColumnTransformer(parts).fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.ct.transform(df), dtype=np.float32)
