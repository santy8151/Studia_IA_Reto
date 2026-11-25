import numpy as np
import pandas as pd

from pfai.data.synthetic_flows import generate_netflow_dataset
from pfai.preprocess import TabularEncoder
from pfai.privacy.audit import full_report
from pfai.privacy.synth import DPTabGAN, benchmark_synthesizers


def test_dptabgan_tracks_privacy_budget_and_decodes_dataframe():
    df = generate_netflow_dataset({"benign": 60, "dos": 30}, seed=11)
    enc = TabularEncoder.from_dataframe(df).fit(df)
    X = enc.transform(df)

    dp = DPTabGAN(
        d_in=X.shape[1],
        d_z=8,
        hidden=32,
        epsilon=5.0,
        delta=1e-3,
        epochs=4,
        batch_size=32,
        encoder=enc,
    ).fit(X, verbose=False)

    budget = dp.privacy_report()
    assert budget["epsilon_spent"] is not None
    assert budget["epsilon_spent"] <= 5.5
    assert budget["delta"] == 1e-3
    assert budget["n_records"] == len(X)

    synth_df = dp.sample_df(40)
    assert isinstance(synth_df, pd.DataFrame)
    assert len(synth_df) == 40
    assert set(synth_df.columns) == set(df.columns)

    rep = full_report(df.iloc[:60], df.iloc[60:], synth_df, label="family")
    assert np.isfinite(rep["mia_auc"])


def test_benchmark_synthesizers_runs_gaussian_copula_and_dp_tabgan():
    df = generate_netflow_dataset({"benign": 60, "dos": 30}, seed=12)
    train_df, holdout_df = df.iloc[:60].reset_index(drop=True), df.iloc[60:].reset_index(drop=True)
    results = benchmark_synthesizers(
        train_df,
        holdout_df,
        label="family",
        methods=("gaussian_copula", "dp_tabgan"),
        dp_epochs=3,
        dp_epsilon=6.0,
        dp_delta=1e-3,
    )
    assert "gaussian_copula" in results
    assert results["gaussian_copula"]["verdict"]["formal_dp"] is False
    assert "dp_tabgan" in results
    assert results["dp_tabgan"]["verdict"]["formal_dp"] is True
    assert results["dp_tabgan"]["privacy_budget"]["epsilon_spent"] > 0
