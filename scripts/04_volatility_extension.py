from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import statsmodels.api as sm
from arch import arch_model
from scipy.stats import kendalltau, spearmanr


def fred_series(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return df.set_index("date")[series_id].dropna().sort_index().resample("ME").last()


def hac_ols(y: pd.Series, x: pd.Series, label: str) -> dict[str, float | int | str]:
    d = pd.concat([y.rename("y"), x.rename("x")], axis=1, join="inner").dropna()
    if len(d) < 40:
        return {"spec": label, "nobs": len(d), "beta": np.nan, "t": np.nan, "p": np.nan, "r2": np.nan}
    X = sm.add_constant(d["x"])
    m = sm.OLS(d["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    return {
        "spec": label,
        "nobs": int(m.nobs),
        "beta": float(m.params["x"]),
        "t": float(m.tvalues["x"]),
        "p": float(m.pvalues["x"]),
        "r2": float(m.rsquared),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "processed" / "volatility_extension"
    out.mkdir(parents=True, exist_ok=True)

    us_mat = sio.loadmat(root / "data" / "input" / "lndata.mat")
    # Original US factor file is external in most setups; expected under data/input/Fhat64.mat if present.
    fhat_path = root / "data" / "input" / "Fhat64.mat"
    if not fhat_path.exists():
        raise FileNotFoundError("Expected data/input/Fhat64.mat for US signal")
    fhat = sio.loadmat(fhat_path)["Fhat_T"]
    us_dates = pd.date_range("1964-01-31", periods=fhat.shape[0], freq="ME")
    us_signal = pd.Series(fhat[:, 0], index=us_dates, name="US_F1").loc["1964-01-31":"2003-12-31"]

    de_factors = pd.read_csv(root / "data" / "processed" / "replication" / "factors.csv", parse_dates=["date"]).set_index("date").sort_index()
    de_signal = de_factors["F1"].rename("DE_F1")
    de_signal.index = de_signal.index.to_period("M").to_timestamp("M")

    us10 = fred_series("GS10")
    de10 = fred_series("IRLTLT01DEM156N")
    us_sq = (us10.diff() ** 2).rename("us_sq")
    de_sq = (de10.diff() ** 2).rename("de_sq")

    am_us = arch_model(us10.diff().dropna() * 100, mean="Zero", vol="GARCH", p=1, q=1)
    us_fit = am_us.fit(disp="off")
    us_garch = ((us_fit.conditional_volatility / 100.0) ** 2).rename("us_garch")

    am_de = arch_model(de10.diff().dropna() * 100, mean="Zero", vol="GARCH", p=1, q=1)
    de_fit = am_de.fit(disp="off")
    de_garch = ((de_fit.conditional_volatility / 100.0) ** 2).rename("de_garch")

    vix = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS")
    vix.columns = ["date", "VIXCLS"]
    vix["date"] = pd.to_datetime(vix["date"])
    vix["VIXCLS"] = pd.to_numeric(vix["VIXCLS"], errors="coerce")
    us_vix = vix.set_index("date")["VIXCLS"].dropna().sort_index().resample("ME").mean().rename("us_vix")

    rows = [
        hac_ols(us_signal, us_sq, "US: F1 ~ sq(y10 change)"),
        hac_ols(us_signal, us_garch, "US: F1 ~ garch var(y10)"),
        hac_ols(us_signal, us_vix, "US: F1 ~ VIX (monthly mean)"),
        hac_ols(de_signal, de_sq, "DE: F1 ~ sq(y10 change)"),
        hac_ols(de_signal, de_garch, "DE: F1 ~ garch var(y10)"),
        hac_ols(us_signal, us_sq.shift(1), "US: F1 ~ lag sq(y10 change)"),
        hac_ols(us_signal, us_garch.shift(1), "US: F1 ~ lag garch var(y10)"),
        hac_ols(us_signal, us_vix.shift(1), "US: F1 ~ lag VIX"),
        hac_ols(de_signal, de_sq.shift(1), "DE: F1 ~ lag sq(y10 change)"),
        hac_ols(de_signal, de_garch.shift(1), "DE: F1 ~ lag garch var(y10)"),
    ]
    reg = pd.DataFrame(rows)

    corr_rows = []
    for label, x, y in [
        ("US: F1 vs sq(y10)", us_sq, us_signal),
        ("US: F1 vs garch(y10)", us_garch, us_signal),
        ("US: F1 vs VIX", us_vix, us_signal),
        ("DE: F1 vs sq(y10)", de_sq, de_signal),
        ("DE: F1 vs garch(y10)", de_garch, de_signal),
    ]:
        d = pd.concat([x.rename("x"), y.rename("y")], axis=1, join="inner").dropna()
        dlag = pd.concat([x.shift(1).rename("x"), y.rename("y")], axis=1, join="inner").dropna()
        srho, sp = spearmanr(d["x"], d["y"]) if len(d) > 20 else (np.nan, np.nan)
        ktau, kp = kendalltau(d["x"], d["y"]) if len(d) > 20 else (np.nan, np.nan)
        corr_rows.append(
            {
                "metric": label,
                "nobs": len(d),
                "corr_contemp": d["x"].corr(d["y"]) if len(d) else np.nan,
                "corr_lag_vol": dlag["x"].corr(dlag["y"]) if len(dlag) else np.nan,
                "spearman_rho": srho,
                "spearman_p": sp,
                "kendall_tau": ktau,
                "kendall_p": kp,
            }
        )
    corr = pd.DataFrame(corr_rows)

    reg.to_csv(out / "vol_signal_regressions.csv", index=False)
    corr.to_csv(out / "vol_signal_correlations_and_rank.csv", index=False)


if __name__ == "__main__":
    main()
