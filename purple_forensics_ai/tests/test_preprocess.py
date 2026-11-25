import numpy as np
import pandas as pd
import pytest

from pfai.data.synthetic_flows import generate_netflow_dataset
from pfai.preprocess import TabularEncoder


def test_generate_netflow_dataset_profiles_and_validation():
    df = generate_netflow_dataset({"benign": 20, "dos": 10, "probe": 5}, seed=1)
    assert len(df) == 35
    assert set(df["family"].unique()) == {"benign", "dos", "probe"}
    with pytest.raises(ValueError, match="desconocida"):
        generate_netflow_dataset({"unknown": 5})


def test_tabular_encoder_roundtrip_and_persistence(tmp_path):
    df = generate_netflow_dataset({"benign": 40, "exfil": 20}, seed=2)
    feat = df.drop(columns=["family"])
    enc = TabularEncoder.from_dataframe(df, exclude=["family"]).fit(feat)

    X = enc.transform(feat)
    assert X.dtype == np.float32
    assert X.shape[0] == len(feat)
    assert len(enc.get_feature_names_out()) == X.shape[1]

    rec = enc.inverse_transform(X)
    assert list(rec.columns) == enc.numeric + enc.categorical
    assert pd.api.types.is_integer_dtype(rec["pkts"].dtype)
    assert (rec["proto"] == feat["proto"]).mean() > 0.99
    assert np.allclose(rec["duration"], feat["duration"], rtol=0.15, atol=0.15)

    saved = enc.save(tmp_path / "enc.pkl")
    loaded = TabularEncoder.load(saved)
    assert np.allclose(loaded.transform(feat), X)


def test_tabular_encoder_handles_nans_and_unknown_categories():
    train = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "cat": ["http", "ssh", "dns", "http"]})
    enc = TabularEncoder(["a"], ["cat"]).fit(train)
    test = pd.DataFrame({"a": [np.nan, np.inf], "cat": ["unknown_proto", None]})
    X = enc.transform(test)
    assert np.isfinite(X).all()
