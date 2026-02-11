"""
Signal utilities for fixed-income alpha construction.

This module is notebook-friendly:
- works with long-format data: one row per (date, product)
- keeps explicit column names
- avoids side effects
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SignalConfig:
    """Configuration for alpha estimation."""

    date_col: str = "time_stamp"
    asset_col: str = "product"
    ret_col: str = "ret_total_per_dv01"
    carry_col: str = "carry_bp_equiv"
    roll_col: str = "roll_ortho_bp"
    mom_window: int = 21
    reversal_window: int = 5
    ewm_span: int = 10
    clip_z: float = 4.0
    w_mom: float = 0.45
    w_carry: float = 0.35
    w_rev: float = 0.10
    w_roll: float = 0.10
    alpha_strength: float = 0.30
    min_std: float = 1e-10


def _cross_sectional_zscore(
    s: pd.Series, clip_z: float = 4.0, min_std: float = 1e-10
) -> pd.Series:
    """Cross-sectional z-score with stable guards."""

    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if pd.isna(sd) or sd < min_std:
        return pd.Series(0.0, index=s.index)
    z = (s - mu) / sd
    return z.clip(-clip_z, clip_z)


def compute_roll_bp_1m(
    data: pd.DataFrame,
    date_col: str = "time_stamp",
    country_col: str = "country_iso",
    maturity_col: str = "remaining_maturity",
    curve_node_col: str = "mat_cat",
    yield_col: str = "yield",
    yield_in_percent: bool = True,
    horizon_months: int = 1,
    out_col: str = "roll_bp_1m",
    clip_bp: Optional[float] = 25.0,
) -> pd.DataFrame:
    """
    Approximate roll-down in bp:
        roll(t, tau, h) = y(t, tau) - y(t, tau-h)
    using linear interpolation on (country, date) curve nodes.
    """

    if horizon_months <= 0:
        raise ValueError("horizon_months must be positive.")

    required = {date_col, country_col, maturity_col, curve_node_col, yield_col}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"Missing columns for roll computation: {sorted(missing)}")

    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[maturity_col] = pd.to_numeric(df[maturity_col], errors="coerce")
    df[curve_node_col] = pd.to_numeric(df[curve_node_col], errors="coerce")
    y = pd.to_numeric(df[yield_col], errors="coerce")
    df["_y_bp_"] = y * 100.0 if yield_in_percent else y * 1e4

    h_year = float(horizon_months) / 12.0

    def _roll_one_group(g: pd.DataFrame) -> pd.DataFrame:
        out = g.copy()
        curve = out.groupby(curve_node_col)["_y_bp_"].median().dropna().sort_index()
        x = curve.index.to_numpy(dtype=float)
        yy = curve.to_numpy(dtype=float)

        if x.size < 2:
            out[out_col] = np.nan
            return out

        tau = out[maturity_col].to_numpy(dtype=float)
        tau_m = tau - h_year

        y_tau = np.interp(np.clip(tau, x.min(), x.max()), x, yy)
        y_tau_m = np.interp(np.clip(tau_m, x.min(), x.max()), x, yy)
        roll = y_tau - y_tau_m
        roll[tau_m < x.min()] = np.nan

        if clip_bp is not None:
            roll = np.clip(roll, -float(clip_bp), float(clip_bp))

        out[out_col] = roll
        return out

    df = df.groupby([date_col, country_col], group_keys=False).apply(_roll_one_group)
    df.drop(columns=["_y_bp_"], inplace=True)
    return df


def orthogonalize_roll_vs_carry(
    data: pd.DataFrame,
    date_col: str = "time_stamp",
    carry_col: str = "carry_bp_equiv",
    roll_col: str = "roll_bp_1m",
    out_col: str = "roll_ortho_bp",
    min_var: float = 1e-12,
) -> pd.DataFrame:
    """
    Remove cross-sectional linear carry component from roll, date by date.
    """

    required = {date_col, carry_col, roll_col}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"Missing columns for roll orthogonalization: {sorted(missing)}")

    df = data.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    def _ortho(g: pd.DataFrame) -> pd.DataFrame:
        out = g.copy()
        c = pd.to_numeric(out[carry_col], errors="coerce")
        r = pd.to_numeric(out[roll_col], errors="coerce")

        c0 = c - c.mean()
        r0 = r - r.mean()
        denom = float((c0 * c0).sum())

        if np.isnan(denom) or denom < min_var:
            out[out_col] = r0.fillna(0.0)
            return out

        beta = float((r0 * c0).sum() / denom)
        out[out_col] = (r - beta * c0).fillna(0.0)
        return out

    return df.groupby(date_col, group_keys=False).apply(_ortho)


def apply_roll_to_signal(
    data: pd.DataFrame,
    base_col: str = "ret_total_per_dv01",
    roll_col: str = "roll_ortho_bp",
    alpha_roll: float = 0.35,
    out_col: str = "ret_total_per_dv01_roll",
) -> pd.DataFrame:
    """Create a new signal by adding roll contribution to base signal."""

    required = {base_col, roll_col}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"Missing columns for signal update: {sorted(missing)}")

    df = data.copy()
    b = pd.to_numeric(df[base_col], errors="coerce").fillna(0.0)
    r = pd.to_numeric(df[roll_col], errors="coerce").fillna(0.0)
    df[out_col] = b + float(alpha_roll) * r
    return df


def build_alpha_panel(data: pd.DataFrame, config: SignalConfig = SignalConfig()) -> pd.DataFrame:
    """
    Build cross-sectional alpha components and combined alpha score.
    """

    required = {config.date_col, config.asset_col, config.ret_col}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"Missing columns for alpha panel: {sorted(missing)}")

    df = data.copy()
    dcol = config.date_col
    acol = config.asset_col
    rcol = config.ret_col
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df[rcol] = pd.to_numeric(df[rcol], errors="coerce").fillna(0.0)

    # Time-series components by asset
    df = df.sort_values([acol, dcol]).reset_index(drop=True)
    g = df.groupby(acol, group_keys=False)[rcol]

    df["mom_raw"] = g.transform(
        lambda x: x.rolling(config.mom_window, min_periods=max(2, config.mom_window // 3))
        .mean()
        .shift(1)
    )
    df["rev_raw"] = -g.transform(
        lambda x: x.rolling(
            config.reversal_window, min_periods=max(2, config.reversal_window // 2)
        )
        .mean()
        .shift(1)
    )

    if config.carry_col in df.columns:
        df["carry_raw"] = pd.to_numeric(df[config.carry_col], errors="coerce")
    else:
        df["carry_raw"] = 0.0

    if config.roll_col in df.columns:
        df["roll_raw"] = pd.to_numeric(df[config.roll_col], errors="coerce")
    else:
        df["roll_raw"] = 0.0

    # Cross-sectional normalization per date
    for src, dst in [
        ("mom_raw", "mom_z"),
        ("rev_raw", "rev_z"),
        ("carry_raw", "carry_z"),
        ("roll_raw", "roll_z"),
    ]:
        df[dst] = df.groupby(dcol)[src].transform(
            lambda s: _cross_sectional_zscore(
                s, clip_z=config.clip_z, min_std=config.min_std
            )
        )

    df["alpha_raw"] = (
        config.w_mom * df["mom_z"].fillna(0.0)
        + config.w_carry * df["carry_z"].fillna(0.0)
        + config.w_rev * df["rev_z"].fillna(0.0)
        + config.w_roll * df["roll_z"].fillna(0.0)
    )

    # Optional smoothing by asset
    df["alpha"] = (
        df.sort_values([acol, dcol])
        .groupby(acol)["alpha_raw"]
        .transform(lambda x: x.ewm(span=config.ewm_span, adjust=False).mean())
    )

    return df


def estimate_expected_returns(
    alpha_panel: pd.DataFrame,
    config: SignalConfig = SignalConfig(),
    as_of: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Convert alpha panel into Markowitz-ready expected returns.

    Returns columns:
      - product
      - ewma_ret_dv01
      - alpha
      - alpha_scaled
    """

    dcol = config.date_col
    acol = config.asset_col
    rcol = config.ret_col
    required = {dcol, acol, rcol, "alpha"}
    missing = required - set(alpha_panel.columns)
    if missing:
        raise KeyError(f"Missing columns for expected returns: {sorted(missing)}")

    df = alpha_panel.copy()
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df[rcol] = pd.to_numeric(df[rcol], errors="coerce").fillna(0.0)
    df["alpha"] = pd.to_numeric(df["alpha"], errors="coerce").fillna(0.0)

    if as_of is None:
        as_of = df[dcol].max()
    as_of = pd.to_datetime(as_of)
    hist = df[df[dcol] <= as_of].copy()

    # Recent EWMA return estimate by asset
    mu_ewm = (
        hist.sort_values([acol, dcol])
        .groupby(acol)[rcol]
        .apply(lambda x: x.ewm(span=max(4, config.ewm_span), adjust=False).mean().iloc[-1])
        .rename("mu_ewm")
    )

    # Latest alpha by asset
    latest = hist.sort_values(dcol).groupby(acol, as_index=False).tail(1)[[acol, "alpha"]]
    latest = latest.set_index(acol)

    out = mu_ewm.to_frame().join(latest, how="left").fillna(0.0)

    # Map alpha (unitless) into return units using cross-sectional scale
    scale = float(out["mu_ewm"].std(ddof=0))
    if not np.isfinite(scale) or scale < config.min_std:
        scale = 1.0
    out["alpha_scaled"] = out["alpha"] * scale
    out["ewma_ret_dv01"] = (
        (1.0 - config.alpha_strength) * out["mu_ewm"]
        + config.alpha_strength * out["alpha_scaled"]
    )

    out = out.reset_index().rename(columns={acol: "product"})
    return out[["product", "ewma_ret_dv01", "alpha", "alpha_scaled"]]


def estimate_expected_returns_from_long(
    data: pd.DataFrame,
    config: SignalConfig = SignalConfig(),
    as_of: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Convenience wrapper: long data -> alpha panel -> expected returns."""

    panel = build_alpha_panel(data, config=config)
    return estimate_expected_returns(panel, config=config, as_of=as_of)


__all__ = [
    "SignalConfig",
    "_cross_sectional_zscore",
    "compute_roll_bp_1m",
    "orthogonalize_roll_vs_carry",
    "apply_roll_to_signal",
    "build_alpha_panel",
    "estimate_expected_returns",
    "estimate_expected_returns_from_long",
]

