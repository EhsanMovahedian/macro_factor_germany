from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import statsmodels.api as sm
from arch import arch_model
from scipy.stats import kendalltau, spearmanr

import investpy


def fred_series(series_id: str, how: str = "last") -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url)
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    s = df.set_index("date")[series_id].dropna().sort_index()
    return s.resample("ME").mean() if how == "mean" else s.resample("ME").last()


def hac_ols(y: pd.Series, x: pd.Series) -> tuple[int, float, float, float, float]:
    d = pd.concat([y.rename("y"), x.rename("x")], axis=1, join="inner").dropna()
    if len(d) < 40:
        return len(d), np.nan, np.nan, np.nan, np.nan
    X = sm.add_constant(d["x"])
    m = sm.OLS(d["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    return int(m.nobs), float(m.params["x"]), float(m.tvalues["x"]), float(m.pvalues["x"]), float(m.rsquared)


def hac_joint_regression(y: pd.Series, X: pd.DataFrame, maxlags: int = 6) -> dict[str, float | int]:
    d = pd.concat([y.rename("y"), X], axis=1, join="inner").dropna()
    if len(d) < 40:
        out = {"nobs": len(d), "r2": np.nan, "adj_r2": np.nan}
        for c in X.columns:
            out[f"b_{c}"] = np.nan
            out[f"t_{c}"] = np.nan
            out[f"p_{c}"] = np.nan
        return out
    Xc = sm.add_constant(d[X.columns], has_constant="add")
    m = sm.OLS(d["y"], Xc).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    out = {"nobs": int(m.nobs), "r2": float(m.rsquared), "adj_r2": float(m.rsquared_adj)}
    for c in X.columns:
        out[f"b_{c}"] = float(m.params[c])
        out[f"t_{c}"] = float(m.tvalues[c])
        out[f"p_{c}"] = float(m.pvalues[c])
    return out


def load_implied_vol(out: Path) -> tuple[pd.Series, pd.Series]:
    cache = out / "de_implied_vol_monthly.csv"
    if cache.exists():
        implied = pd.read_csv(cache, parse_dates=["date"]).set_index("date").sort_index()
        return implied["de_vdax"].rename("de_vdax"), implied["de_vstoxx"].rename("de_vstoxx")

    vdax_daily = investpy.get_index_historical_data(
        index="VDAX New", country="germany", from_date="01/01/2000", to_date="31/12/2022"
    )
    vdax = vdax_daily["Close"].astype(float)
    vdax.index = pd.to_datetime(vdax.index)
    vdax = vdax.resample("ME").mean().rename("de_vdax")

    vstoxx_daily = investpy.get_index_historical_data(
        index="STOXX 50 Volatility VSTOXX EUR",
        country="euro zone",
        from_date="01/01/2000",
        to_date="31/12/2022",
    )
    vstoxx = vstoxx_daily["Close"].astype(float)
    vstoxx.index = pd.to_datetime(vstoxx.index)
    vstoxx = vstoxx.resample("ME").mean().rename("de_vstoxx")

    pd.concat([vdax, vstoxx], axis=1).to_csv(cache, index_label="date")
    return vdax, vstoxx


def standardize_series(s: pd.Series) -> pd.Series:
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=1)
    if pd.isna(sd) or sd == 0:
        return s * np.nan
    return (s - mu) / sd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "data" / "processed" / "volatility_extension"
    out.mkdir(parents=True, exist_ok=True)

    # US factors (F1..F5) from original replication file
    fhat_path = root / "data" / "input" / "Fhat64.mat"
    if not fhat_path.exists():
        raise FileNotFoundError("Expected data/input/Fhat64.mat for US factors")
    us_fhat = sio.loadmat(fhat_path)["Fhat_T"]
    us_dates = pd.date_range("1964-01-31", periods=us_fhat.shape[0], freq="ME")
    us_fac = pd.DataFrame(us_fhat[:, :5], index=us_dates, columns=[f"F{i}" for i in range(1, 6)])
    us_fac = us_fac.loc["1964-01-31":"2003-12-31"]

    # Germany factors (F1..F5)
    de_fac = pd.read_csv(
        root / "data" / "processed" / "replication" / "factors.csv", parse_dates=["date"]
    ).set_index("date").sort_index()
    de_fac = de_fac[[f"F{i}" for i in range(1, 6)]].copy()
    de_fac.index = de_fac.index.to_period("M").to_timestamp("M")

    # Vol proxies from rates
    us10 = fred_series("GS10", "last")
    de10 = fred_series("IRLTLT01DEM156N", "last")
    us_sq = (us10.diff() ** 2).rename("us_sq")
    de_sq = (de10.diff() ** 2).rename("de_sq")

    us_garch = (
        (arch_model(us10.diff().dropna() * 100, mean="Zero", vol="GARCH", p=1, q=1).fit(disp="off").conditional_volatility / 100.0)
        ** 2
    ).rename("us_garch")
    de_garch = (
        (arch_model(de10.diff().dropna() * 100, mean="Zero", vol="GARCH", p=1, q=1).fit(disp="off").conditional_volatility / 100.0)
        ** 2
    ).rename("de_garch")

    # US implied vol
    us_vix = fred_series("VIXCLS", "mean").rename("us_vix")

    # Germany / Euro implied vol proxies
    vdax, vstoxx = load_implied_vol(out)

    rows_reg = []
    rows_rank = []

    for country, fac_df, proxies in [
        ("US", us_fac, {"sq": us_sq, "garch": us_garch, "vix": us_vix}),
        ("DE", de_fac, {"sq": de_sq, "garch": de_garch, "vdax": vdax, "vstoxx": vstoxx}),
    ]:
        for f in fac_df.columns:
            y = fac_df[f]
            for pname, x in proxies.items():
                n, b, t, p, r2 = hac_ols(y, x)
                rows_reg.append({"country": country, "factor": f, "proxy": pname, "lag": 0, "nobs": n, "beta": b, "t": t, "p": p, "r2": r2})

                n, b, t, p, r2 = hac_ols(y, x.shift(1))
                rows_reg.append({"country": country, "factor": f, "proxy": pname, "lag": 1, "nobs": n, "beta": b, "t": t, "p": p, "r2": r2})

                d = pd.concat([y.rename("y"), x.rename("x")], axis=1, join="inner").dropna()
                if len(d) >= 20:
                    srho, sp = spearmanr(d["x"], d["y"])
                    ktau, kp = kendalltau(d["x"], d["y"])
                else:
                    srho = sp = ktau = kp = np.nan
                rows_rank.append(
                    {
                        "country": country,
                        "factor": f,
                        "proxy": pname,
                        "nobs": len(d),
                        "spearman_rho": srho,
                        "spearman_p": sp,
                        "kendall_tau": ktau,
                        "kendall_p": kp,
                    }
                )

    reg = pd.DataFrame(rows_reg)
    rank = pd.DataFrame(rows_rank)

    reg.to_csv(out / "vol_signal_regressions_F1_F5_with_de_implied_vol.csv", index=False)
    rank.to_csv(out / "vol_signal_rank_tests_F1_F5_with_de_implied_vol.csv", index=False)

    obs_rows = []
    for f in us_fac.columns:
        for pn, x in {"sq": us_sq, "garch": us_garch, "vix": us_vix}.items():
            obs_rows.append({"country": "US", "factor": f, "proxy": pn, "nobs": len(pd.concat([us_fac[f], x], axis=1, join="inner").dropna())})
    for f in de_fac.columns:
        for pn, x in {"sq": de_sq, "garch": de_garch, "vdax": vdax, "vstoxx": vstoxx}.items():
            obs_rows.append({"country": "DE", "factor": f, "proxy": pn, "nobs": len(pd.concat([de_fac[f], x], axis=1, join="inner").dropna())})
    pd.DataFrame(obs_rows).to_csv(out / "observation_counts_F1_F5_with_de_implied_vol.csv", index=False)

    horizons = [0, 1, 3, 6, 12]
    joint_rows: list[dict[str, float | int | str]] = []
    for country, X, proxies in [
        ("US", us_fac, {"sq": us_sq, "garch": us_garch, "vix": us_vix}),
        ("DE", de_fac, {"sq": de_sq, "garch": de_garch, "vdax": vdax.rename("vdax"), "vstoxx": vstoxx.rename("vstoxx")}),
    ]:
        for proxy_name, proxy in proxies.items():
            for h in horizons:
                y = proxy if h == 0 else proxy.shift(-h)
                res = hac_joint_regression(y, X)
                joint_rows.append({"country": country, "proxy": proxy_name, "horizon_m": h, **res})
    pd.DataFrame(joint_rows).to_csv(out / "multifactor_proxy_regressions_horizons.csv", index=False)

    # Standardized-proxy versions for comparability checks.
    us_sq_z = standardize_series(us_sq).rename("us_sq")
    us_garch_z = standardize_series(us_garch).rename("us_garch")
    us_vix_z = standardize_series(us_vix).rename("us_vix")
    de_sq_z = standardize_series(de_sq).rename("de_sq")
    de_garch_z = standardize_series(de_garch).rename("de_garch")
    vdax_z = standardize_series(vdax).rename("de_vdax")
    vstoxx_z = standardize_series(vstoxx).rename("de_vstoxx")

    rows_reg_std = []
    rows_rank_std = []
    for country, fac_df, proxies in [
        ("US", us_fac, {"sq": us_sq_z, "garch": us_garch_z, "vix": us_vix_z}),
        ("DE", de_fac, {"sq": de_sq_z, "garch": de_garch_z, "vdax": vdax_z, "vstoxx": vstoxx_z}),
    ]:
        for f in fac_df.columns:
            y = fac_df[f]
            for pname, x in proxies.items():
                n, b, t, p, r2 = hac_ols(y, x)
                rows_reg_std.append({"country": country, "factor": f, "proxy": pname, "lag": 0, "nobs": n, "beta": b, "t": t, "p": p, "r2": r2})

                n, b, t, p, r2 = hac_ols(y, x.shift(1))
                rows_reg_std.append({"country": country, "factor": f, "proxy": pname, "lag": 1, "nobs": n, "beta": b, "t": t, "p": p, "r2": r2})

                d = pd.concat([y.rename("y"), x.rename("x")], axis=1, join="inner").dropna()
                if len(d) >= 20:
                    srho, sp = spearmanr(d["x"], d["y"])
                    ktau, kp = kendalltau(d["x"], d["y"])
                else:
                    srho = sp = ktau = kp = np.nan
                rows_rank_std.append(
                    {
                        "country": country,
                        "factor": f,
                        "proxy": pname,
                        "nobs": len(d),
                        "spearman_rho": srho,
                        "spearman_p": sp,
                        "kendall_tau": ktau,
                        "kendall_p": kp,
                    }
                )

    pd.DataFrame(rows_reg_std).to_csv(out / "vol_signal_regressions_F1_F5_with_de_implied_vol_std_proxies.csv", index=False)
    pd.DataFrame(rows_rank_std).to_csv(out / "vol_signal_rank_tests_F1_F5_with_de_implied_vol_std_proxies.csv", index=False)

    joint_rows_std: list[dict[str, float | int | str]] = []
    for country, X, proxies in [
        ("US", us_fac, {"sq": us_sq_z, "garch": us_garch_z, "vix": us_vix_z}),
        ("DE", de_fac, {"sq": de_sq_z, "garch": de_garch_z, "vdax": vdax_z.rename("vdax"), "vstoxx": vstoxx_z.rename("vstoxx")}),
    ]:
        for proxy_name, proxy in proxies.items():
            for h in horizons:
                y = proxy if h == 0 else proxy.shift(-h)
                res = hac_joint_regression(y, X)
                joint_rows_std.append({"country": country, "proxy": proxy_name, "horizon_m": h, **res})
    pd.DataFrame(joint_rows_std).to_csv(out / "multifactor_proxy_regressions_horizons_std_proxies.csv", index=False)

    # Optional: keep backward-compatible simpler exports if desired by downstream docs
    # (using F1 rows only)
    f1_reg = reg[(reg["factor"] == "F1") & (reg["proxy"].isin(["sq", "garch", "vix"])) & (reg["country"] == "US") | (reg["factor"] == "F1") & (reg["proxy"].isin(["sq", "garch"])) & (reg["country"] == "DE")]
    f1_rank = rank[(rank["factor"] == "F1") & (((rank["country"] == "US") & (rank["proxy"].isin(["sq", "garch", "vix"]))) | ((rank["country"] == "DE") & (rank["proxy"].isin(["sq", "garch"]))))]
    f1_reg.to_csv(out / "vol_signal_regressions.csv", index=False)
    f1_rank[["country", "factor", "proxy", "nobs", "spearman_rho", "spearman_p", "kendall_tau", "kendall_p"]].to_csv(
        out / "vol_signal_nonlinear_rank_stats.csv", index=False
    )


if __name__ == "__main__":
    main()
