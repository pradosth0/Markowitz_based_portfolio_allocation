"""
PortfolioDashboardVisualizer — Bloomberg Dark Terminal Style
============================================================
Version 3 — polices agrandies + nouveaux plots :
  - Rolling Sharpe 6M
  - Distribution des returns (histogram + VaR/CVaR)
  - PnL cumulé par type de produit
  - Exposition par pays dans le temps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")


# ─── PALETTE ──────────────────────────────────────────────────────────────────

DARK = {
    "bg":      "#07090F",
    "surface": "#0C1120",
    "border":  "#1C2840",
    "accent":  "#1A7FFF",
    "green":   "#00D49A",
    "red":     "#FF3D5A",
    "gold":    "#F0A500",
    "text":    "#D8E4F0",
    "muted":   "#4E6080",
    "dim":     "#233050",
    "grid":    "#111A2C",
}

ASSET_COLORS = [
    "#1A7FFF","#00D49A","#F0A500","#FF3D5A",
    "#7C5CFC","#00C4D4","#F87C00","#22C97A",
    "#E040FB","#546E8A","#A3D900","#FF7043",
]

RYGCMAP = LinearSegmentedColormap.from_list(
    "ryg", ["#FF3D5A","#111A2C","#00D49A"], N=256
)

# Tailles de police — tout est défini ici pour modifier facilement
FS = {
    "header_title":  22,
    "header_sub":    13,
    "header_nav":    16,
    "kpi_label":     11,
    "kpi_value":     18,
    "section_bar":   20,
    "section_title": 13,
    "section_badge": 12,
    "axis_label":    11,
    "tick":          10,
    "legend":        11,
    "annotation":    11,
    "heatmap_cell":  11,
}


# ─── STYLE GLOBAL ─────────────────────────────────────────────────────────────

def _apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  DARK["bg"],
        "axes.facecolor":    DARK["surface"],
        "axes.edgecolor":    DARK["border"],
        "axes.linewidth":    0.8,
        "axes.labelcolor":   DARK["muted"],
        "axes.labelsize":    FS["axis_label"],
        "axes.titlesize":    FS["section_title"],
        "axes.titlecolor":   DARK["text"],
        "xtick.color":       DARK["muted"],
        "ytick.color":       DARK["muted"],
        "xtick.labelsize":   FS["tick"],
        "ytick.labelsize":   FS["tick"],
        "xtick.major.size":  0,
        "ytick.major.size":  0,
        "grid.color":        DARK["grid"],
        "grid.linewidth":    0.5,
        "grid.linestyle":    "--",
        "grid.alpha":        1.0,
        "legend.facecolor":  DARK["surface"],
        "legend.edgecolor":  DARK["border"],
        "legend.labelcolor": DARK["muted"],
        "legend.fontsize":   FS["legend"],
        "font.family":       "monospace",
        "text.color":        DARK["text"],
    })


# ─── FORMATTERS ───────────────────────────────────────────────────────────────

def _fmt_pct(v, d=1):
    return f"{v*100:+.{d}f}%" if (v is not None and not np.isnan(v)) else "—"

def _fmt_pct_plain(v, d=1):
    return f"{v*100:.{d}f}%" if (v is not None and not np.isnan(v)) else "—"

def _fmt_cur(v):
    if abs(v) >= 1e6: return f"{v/1e6:.2f}M€"
    if abs(v) >= 1e3: return f"{v/1e3:.1f}K€"
    return f"{v:.0f}€"

def _pct_fmt(v, _=None):  return f"{v*100:.1f}%"
def _cur_fmt(v, _=None):  return _fmt_cur(v)
def _color_s(v):          return DARK["green"] if v >= 0 else DARK["red"]


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor(DARK["surface"])
    ax.tick_params(colors=DARK["muted"], which="both", labelsize=FS["tick"])
    for sp in ax.spines.values():
        sp.set_color(DARK["border"])
    ax.grid(True, color=DARK["grid"], linewidth=0.5, linestyle="--", alpha=0.9)


def _section_title(ax, title, badge=None):
    ax.text(-0.01, 1.05, "▌", transform=ax.transAxes,
            fontsize=FS["section_bar"], color=DARK["accent"],
            va="bottom", ha="right", clip_on=False, fontweight="bold")
    ax.text(0.005, 1.05, title.upper(), transform=ax.transAxes,
            fontsize=FS["section_title"], color=DARK["text"],
            va="bottom", ha="left", clip_on=False,
            fontweight="bold", fontfamily="monospace")
    if badge:
        ax.text(0.005 + len(title)*0.011 + 0.02, 1.05, f"· {badge}",
                transform=ax.transAxes, fontsize=FS["section_badge"],
                color=DARK["accent"], va="bottom", ha="left",
                clip_on=False, fontfamily="monospace")


# ─── HEADER ───────────────────────────────────────────────────────────────────

def _draw_header(fig, metrics, capital):
    final_val  = metrics.get("final_value", capital)
    total_ret  = metrics.get("total_return", 0) or 0
    ann_ret    = metrics.get("annual_return", 0) or 0
    sharpe     = metrics.get("sharpe", 0) or 0
    max_dd     = metrics.get("max_drawdown", 0) or 0

    fig.text(0.012, 0.988, "▌", fontsize=36, color=DARK["accent"], va="top")
    fig.text(0.025, 0.989, "PORTFOLIO ANALYTICS",
             fontsize=FS["header_title"], color=DARK["text"],
             fontweight="bold", fontfamily="monospace", va="top")
    fig.text(0.025, 0.978, "Fixed Income  ·  Multi-Product  ·  DV01-Normalized",
             fontsize=FS["header_sub"], color=DARK["muted"],
             fontfamily="monospace", va="top")

    kpis_h = [
        ("NAV",          _fmt_cur(final_val),       _color_s(final_val - capital)),
        ("TOTAL RETURN", _fmt_pct(total_ret),        _color_s(total_ret)),
        ("ANN. RETURN",  _fmt_pct(ann_ret),          _color_s(ann_ret)),
        ("SHARPE",       f"{sharpe:.2f}",
                         DARK["green"] if sharpe > 1 else DARK["gold"]),
        ("MAX DD",       _fmt_pct(max_dd),           DARK["red"]),
    ]
    for i, (lbl, val, col) in enumerate(kpis_h):
        x = 0.58 + i * 0.085
        fig.text(x, 0.987, lbl, fontsize=FS["kpi_label"] - 1,
                 color=DARK["muted"], fontfamily="monospace", va="top")
        fig.text(x, 0.977, val, fontsize=FS["header_nav"],
                 color=col, fontweight="bold", fontfamily="monospace", va="top")

    fig.add_artist(plt.Line2D(
        [0.01, 0.99], [0.968, 0.968], transform=fig.transFigure,
        color=DARK["border"], linewidth=1.0
    ))


# ─── KPI PANEL ────────────────────────────────────────────────────────────────

def _draw_kpi_panel(ax, metrics):
    ax.set_facecolor(DARK["dim"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    sharpe   = metrics.get("sharpe", 0) or 0
    calmar   = metrics.get("calmar_ratio", 0) or 0
    win_rate = metrics.get("win_rate", 0) or 0
    var_95   = metrics.get("var_95", 0) or 0
    cvar_95  = metrics.get("cvar_95", 0) or 0
    vol      = metrics.get("volatility", 0) or 0
    tc_total = metrics.get("tc_total", 0) or 0
    total_pnl= metrics.get("total_pnl", 0) or 0
    tc_ratio = tc_total / max(abs(total_pnl), 1)

    kpis = [
        ("VOLATILITY",  _fmt_pct_plain(vol),           DARK["text"]),
        ("SHARPE",      f"{sharpe:.2f}",               DARK["green"] if sharpe > 1 else DARK["gold"]),
        ("CALMAR",      f"{calmar:.2f}",               DARK["green"] if calmar > 1 else DARK["gold"]),
        ("WIN RATE",    _fmt_pct_plain(win_rate),      DARK["text"]),
        ("VaR 95%",     _fmt_pct(var_95),              DARK["red"]),
        ("CVaR 95%",    _fmt_pct(cvar_95),             DARK["red"]),
        ("TOTAL P&L",   _fmt_cur(total_pnl),           _color_s(total_pnl)),
        ("TC TOTAL",    _fmt_cur(tc_total),            DARK["gold"]),
        ("TC / PNL",    f"{tc_ratio:.1%}",             DARK["gold"]),
    ]

    n  = len(kpis)
    xs = np.linspace(0, 1, n + 1)
    for i, (label, value, color) in enumerate(kpis):
        cx = (xs[i] + xs[i+1]) / 2
        ax.text(cx, 0.78, label, ha="center", va="center",
                fontsize=FS["kpi_label"], color=DARK["muted"],
                fontfamily="monospace", transform=ax.transAxes)
        ax.text(cx, 0.28, value, ha="center", va="center",
                fontsize=FS["kpi_value"], color=color,
                fontfamily="monospace", fontweight="bold",
                transform=ax.transAxes)
        if i < n - 1:
            ax.axvline(xs[i+1], color=DARK["border"], linewidth=0.8, alpha=0.6)


# ─── 1. CUMULATIVE RETURN ─────────────────────────────────────────────────────

def _plot_cumulative_return(ax, pf_cum, bm_cum=None):
    _style_ax(ax)
    badge = _fmt_pct(float(pf_cum.iloc[-1])) if len(pf_cum) else ""
    _section_title(ax, "Cumulative Return", badge)

    x = pf_cum.index
    ax.plot(x, pf_cum.values, color=DARK["accent"], lw=2.5,
            label="Portfolio", zorder=4)
    ax.fill_between(x, pf_cum.values, 0, color=DARK["accent"], alpha=0.10)

    if bm_cum is not None and len(bm_cum):
        ax.plot(x, bm_cum.values, color=DARK["gold"], lw=1.6,
                linestyle="--", alpha=0.8, label="Benchmark", zorder=3)
        ax.fill_between(x, pf_cum.values, bm_cum.values,
                        where=pf_cum.values >= bm_cum.values,
                        color=DARK["green"], alpha=0.12, interpolate=True)
        ax.fill_between(x, pf_cum.values, bm_cum.values,
                        where=pf_cum.values < bm_cum.values,
                        color=DARK["red"], alpha=0.12, interpolate=True)
        ax.annotate(_fmt_pct(float(bm_cum.iloc[-1])),
                    xy=(bm_cum.index[-1], bm_cum.iloc[-1]),
                    xytext=(-60, -16), textcoords="offset points",
                    fontsize=FS["annotation"], color=DARK["gold"],
                    arrowprops=dict(arrowstyle="-", color=DARK["muted"], lw=0.6))

    ax.axhline(0, color=DARK["border"], lw=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
    ax.tick_params(axis="x", rotation=25, labelsize=FS["tick"])
    ax.legend(loc="upper left", framealpha=0.85, fontsize=FS["legend"])
    ax.annotate(_fmt_pct(float(pf_cum.iloc[-1])),
                xy=(pf_cum.index[-1], pf_cum.iloc[-1]),
                xytext=(-60, 12), textcoords="offset points",
                fontsize=FS["annotation"] + 1, color=DARK["accent"],
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=DARK["muted"], lw=0.6))


# ─── 2. DRAWDOWN ──────────────────────────────────────────────────────────────

def _plot_drawdown(ax, dd_series, max_dd=None):
    _style_ax(ax)
    badge = f"Max {_fmt_pct(max_dd)}" if max_dd is not None else ""
    _section_title(ax, "Drawdown", badge)

    x = dd_series.index
    ax.fill_between(x, dd_series.values, 0,
                    color=DARK["red"], alpha=0.35, zorder=2)
    ax.plot(x, dd_series.values, color=DARK["red"], lw=1.5, zorder=3)
    ax.axhline(0, color=DARK["border"], lw=0.8)

    if max_dd is not None and not np.isnan(max_dd):
        idx_min = dd_series.idxmin()
        ax.annotate(_fmt_pct(max_dd),
                    xy=(idx_min, max_dd),
                    xytext=(30, -20), textcoords="offset points",
                    fontsize=FS["annotation"] + 1, color=DARK["red"],
                    fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=DARK["muted"], lw=0.6))

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    ax.tick_params(axis="x", rotation=25, labelsize=FS["tick"])


# ─── 3. ROLLING SHARPE ────────────────────────────────────────────────────────

def _plot_rolling_sharpe(ax, portfolio_returns, window_months=6):
    """Rolling Sharpe sur fenêtre glissante — détecte les régimes."""
    _style_ax(ax)
    window = window_months * 21
    roll   = portfolio_returns.rolling(window)
    rs     = roll.mean() / roll.std() * np.sqrt(252)
    rs     = rs.dropna()

    _section_title(ax, f"Rolling Sharpe · {window_months}M",
                   f"Current {rs.iloc[-1]:.2f}" if len(rs) else "")

    # Zone colorée selon signe
    ax.fill_between(rs.index, rs.values, 0,
                    where=rs.values >= 0, color=DARK["green"], alpha=0.18)
    ax.fill_between(rs.index, rs.values, 0,
                    where=rs.values < 0,  color=DARK["red"],   alpha=0.18)
    ax.plot(rs.index, rs.values, color=DARK["accent"], lw=2.0, zorder=4)

    # Lignes de référence
    for level, lbl, col in [(1.0, "Sharpe=1", DARK["green"]),
                             (0.0, "",         DARK["border"]),
                             (-1.0,"Sharpe=-1",DARK["red"])]:
        ax.axhline(level, color=col, lw=0.9, linestyle="--", alpha=0.7)
        if lbl:
            ax.text(rs.index[0], level + 0.05, lbl,
                    fontsize=FS["annotation"] - 1, color=col, alpha=0.8)

    ax.tick_params(axis="x", rotation=25, labelsize=FS["tick"])
    ax.set_ylabel("Sharpe", fontsize=FS["axis_label"])


# ─── 4. RETURN DISTRIBUTION ───────────────────────────────────────────────────

def _plot_return_distribution(ax, portfolio_returns):
    """Histogramme des returns journaliers + VaR + CVaR."""
    _style_ax(ax)
    ax.grid(True, axis="x", color=DARK["grid"], linewidth=0.5)
    ax.grid(False, axis="y")
    _section_title(ax, "Return Distribution")

    r    = portfolio_returns.dropna() * 100   # en %
    var  = np.percentile(r, 5)
    cvar = r[r <= var].mean()

    # Histogramme
    n_bins = min(60, max(20, len(r) // 15))
    counts, bins, patches = ax.hist(
        r, bins=n_bins, color=DARK["accent"],
        alpha=0.65, edgecolor=DARK["surface"], linewidth=0.3
    )

    # Colorer les barres en queue gauche
    for patch, left in zip(patches, bins[:-1]):
        if left <= var:
            patch.set_facecolor(DARK["red"])
            patch.set_alpha(0.80)

    # VaR et CVaR
    ymax = counts.max()
    ax.axvline(var, color=DARK["red"], lw=2.0, linestyle="--", zorder=5)
    ax.axvline(cvar, color=DARK["gold"], lw=1.5, linestyle=":", zorder=5)
    ax.text(var  - 0.02, ymax * 0.88, f"VaR 95%\n{var:.2f}%",
            ha="right", fontsize=FS["annotation"], color=DARK["red"],
            fontweight="bold")
    ax.text(cvar - 0.02, ymax * 0.65, f"CVaR 95%\n{cvar:.2f}%",
            ha="right", fontsize=FS["annotation"], color=DARK["gold"])

    # Ligne normale théorique
    from scipy.stats import norm as _norm
    mu_r, std_r = r.mean(), r.std()
    xg = np.linspace(r.min(), r.max(), 200)
    bin_w = (bins[-1] - bins[0]) / n_bins
    ax.plot(xg, _norm.pdf(xg, mu_r, std_r) * len(r) * bin_w,
            color=DARK["muted"], lw=1.5, linestyle="-", alpha=0.7,
            label="Normal fit")
    ax.legend(loc="upper right", fontsize=FS["legend"])
    ax.set_xlabel("Daily Return (%)", fontsize=FS["axis_label"])
    ax.tick_params(labelsize=FS["tick"])


# ─── 5. PNL PAR TYPE DE PRODUIT ───────────────────────────────────────────────

def _plot_pnl_by_type(ax, ts_portfolio):
    """
    PnL cumulé par type de produit (bond / spread_country / spread_curve).
    Requiert ts_portfolio.data et ts_portfolio.weights_history.
    """
    _style_ax(ax)
    _section_title(ax, "Cumulative PnL by Product Type")

    try:
        data     = ts_portfolio.data
        wh       = ts_portfolio.weights_history
        capital  = getattr(ts_portfolio, "capital_init", 100_000)
        all_dates= pd.DatetimeIndex(data["time_stamp"].unique()).sort_values()

        # Types disponibles
        type_map = data.groupby("product")["type"].first().to_dict()
        types    = ["bond", "spread_country", "spread_curve"]
        colors_t = [DARK["accent"], DARK["green"], DARK["gold"]]

        pnl_pivot = data.pivot_table(
            index="time_stamp", columns="product",
            values="pnl_total_unit", aggfunc="last"
        )

        reb_dates = wh.index
        pnl_by_type = {t: pd.Series(0.0, index=all_dates) for t in types}

        for i, t_reb in enumerate(reb_dates):
            t_next = reb_dates[i+1] if i+1 < len(reb_dates) else all_dates[-1]
            w_row  = wh.iloc[i]
            mask   = (all_dates > t_reb) & (all_dates <= t_next)
            dates_slice = all_dates[mask]

            for prod, w in w_row.items():
                ptype = type_map.get(prod, "other")
                if ptype not in pnl_by_type:
                    continue
                if prod not in pnl_pivot.columns:
                    continue
                pnl_series = pnl_pivot[prod].reindex(dates_slice).fillna(0)
                pnl_by_type[ptype].loc[dates_slice] += (
                    capital * w * pnl_series
                )

        for ptype, col in zip(types, colors_t):
            s = pnl_by_type[ptype].cumsum().reindex(all_dates).fillna(method="ffill")
            ax.plot(s.index, s.values, color=col, lw=2.0,
                    label=ptype.replace("_", " ").title())
            ax.fill_between(s.index, s.values, 0,
                            color=col, alpha=0.07)

        ax.axhline(0, color=DARK["border"], lw=0.8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_cur_fmt))
        ax.legend(loc="upper left", fontsize=FS["legend"], framealpha=0.9)
        ax.tick_params(axis="x", rotation=25, labelsize=FS["tick"])

    except Exception as e:
        ax.text(0.5, 0.5, f"PnL by type\nnot available\n{e}",
                ha="center", va="center", transform=ax.transAxes,
                color=DARK["muted"], fontsize=FS["annotation"])


# ─── 6. EXPOSITION PAR PAYS ───────────────────────────────────────────────────

def _plot_country_exposure(ax, ts_portfolio):
    """Exposition nette par pays dans le temps (depuis weights_history)."""
    _style_ax(ax)
    _section_title(ax, "Net Exposure by Country")

    try:
        expo   = ts_portfolio.exposition_by_country()
        # Normaliser par somme abs pour avoir en % du portefeuille
        total  = expo.abs().sum(axis=1).replace(0, np.nan)
        expo_n = expo.div(total, axis=0).fillna(0)

        colors_c = {"FR": DARK["accent"], "DE": DARK["green"], "IT": DARK["red"]}

        for col in expo_n.columns:
            col_color = colors_c.get(col, DARK["gold"])
            ax.plot(expo_n.index, expo_n[col].values,
                    color=col_color, lw=2.0, label=col)
            ax.fill_between(expo_n.index, expo_n[col].values, 0,
                            color=col_color, alpha=0.10)

        ax.axhline(0, color=DARK["border"], lw=0.8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_fmt))
        ax.legend(loc="upper right", fontsize=FS["legend"], framealpha=0.9)
        ax.tick_params(axis="x", rotation=25, labelsize=FS["tick"])

    except Exception as e:
        ax.text(0.5, 0.5, f"Country exposure\nnot available\n{e}",
                ha="center", va="center", transform=ax.transAxes,
                color=DARK["muted"], fontsize=FS["annotation"])


# ─── 7. MONTHLY P&L WATERFALL ─────────────────────────────────────────────────

def _plot_monthly_pnl(ax, portfolio_value, capital):
    _style_ax(ax)
    monthly  = portfolio_value.resample("ME").last().ffill()
    cum_pnl  = monthly - capital
    month_ret = cum_pnl.diff().fillna(cum_pnl.iloc[0])
    bottoms  = cum_pnl.shift(1).fillna(0)

    _section_title(ax, "Monthly P&L (Cumulative)",
                   _fmt_cur(float(cum_pnl.iloc[-1])))

    colors = [DARK["green"] if v >= 0 else DARK["red"] for v in month_ret]
    ax.bar(monthly.index, month_ret.values, bottom=bottoms.values,
           color=colors, alpha=0.80, width=20,
           edgecolor=DARK["surface"], linewidth=0.4)
    ax.plot(cum_pnl.index, cum_pnl.values, color=DARK["text"],
            lw=2.0, marker="o", markersize=3.0, zorder=4)
    ax.axhline(0, color=DARK["border"], lw=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_cur_fmt))
    ax.tick_params(axis="x", rotation=35, labelsize=FS["tick"])


# ─── 8. MONTHLY HEATMAP ───────────────────────────────────────────────────────

def _plot_monthly_heatmap(ax, portfolio_returns):
    _style_ax(ax)
    _section_title(ax, "Monthly Returns Heatmap")

    monthly = (1 + portfolio_returns.fillna(0)).resample("ME").prod() - 1
    monthly.index = pd.to_datetime(monthly.index)

    years  = sorted(monthly.index.year.unique())
    months = range(1, 13)
    mlbls  = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

    mat = np.full((len(years), 12), np.nan)
    for i, y in enumerate(years):
        for j, m in enumerate(months):
            mask = (monthly.index.year == y) & (monthly.index.month == m)
            if mask.any():
                mat[i, j] = monthly[mask].values[0] * 100

    vmax = np.nanpercentile(np.abs(mat), 90) or 1.0
    ax.imshow(mat, aspect="auto", cmap=RYGCMAP,
              vmin=-vmax, vmax=vmax, interpolation="nearest")

    ax.set_xticks(range(12))
    ax.set_xticklabels(mlbls, fontsize=FS["tick"] + 1, color=DARK["muted"])
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels([str(y) for y in years],
                       fontsize=FS["tick"] + 1, color=DARK["muted"])
    ax.tick_params(length=0)
    ax.grid(False)

    for i in range(len(years)):
        for j in range(12):
            v = mat[i, j]
            if not np.isnan(v):
                txt_col = DARK["text"] if abs(v) < vmax * 0.6 else DARK["bg"]
                ax.text(j, i, f"{v:+.2f}%", ha="center", va="center",
                        fontsize=FS["heatmap_cell"], color=txt_col,
                        fontfamily="monospace",
                        fontweight="bold" if abs(v) > vmax * 0.5 else "normal")


# ─── 9. WEIGHTS COMPOSITION ───────────────────────────────────────────────────

def _plot_weights_composition(ax, weights_history, top_n=8):
    _style_ax(ax)
    _section_title(ax, f"Portfolio Composition · Top {top_n}")

    avg      = weights_history.abs().mean()
    top_cols = avg.nlargest(top_n).index.tolist()
    df       = weights_history[top_cols].abs().copy()
    others   = weights_history.abs().drop(columns=top_cols, errors="ignore").sum(axis=1)
    if others.max() > 0.001:
        df["Others"] = others

    colors = (ASSET_COLORS * 4)[:len(df.columns)]
    ax.stackplot(df.index, df.T.values, labels=df.columns,
                 colors=colors, alpha=0.82)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.tick_params(axis="x", rotation=25, labelsize=FS["tick"])
    ax.legend(loc="upper left", ncol=2, fontsize=FS["legend"] - 1,
              framealpha=0.9, labelcolor=DARK["text"])


# ─── 10. TC BREAKDOWN ────────────────────────────────────────────────────────

def _plot_tc_breakdown(ax, metrics):
    _style_ax(ax)
    tc_mkt  = metrics.get("transaction_costs_markowitz")
    tc_roll = metrics.get("transaction_costs_rolldown")
    tc_tot  = metrics.get("transaction_costs")

    if tc_mkt is None or tc_roll is None:
        if tc_tot is not None:
            _section_title(ax, "Transaction Costs",
                           _fmt_cur(float(tc_tot.sum())))
            ax.bar(tc_tot.index, tc_tot.values,
                   color=DARK["gold"], alpha=0.8, width=20,
                   edgecolor=DARK["surface"], linewidth=0.4)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(_cur_fmt))
            ax.tick_params(axis="x", rotation=30, labelsize=FS["tick"])
        return

    total_tc = float(tc_mkt.sum() + tc_roll.sum())
    _section_title(ax, "Transaction Costs — Markowitz vs Roll-Down",
                   f"Total {_fmt_cur(total_tc)}")

    x = tc_mkt.index
    w = max((x[1] - x[0]).days * 0.4 if len(x) > 1 else 15, 10)
    ax.bar(x, tc_mkt.values, width=w, label="Markowitz rebalance",
           color=DARK["accent"], alpha=0.82,
           edgecolor=DARK["surface"], linewidth=0.4)
    ax.bar(x, tc_roll.values, width=w, bottom=tc_mkt.values,
           label="Roll-down", color=DARK["gold"], alpha=0.82,
           edgecolor=DARK["surface"], linewidth=0.4)

    ax2 = ax.twinx()
    cum = (tc_mkt + tc_roll).cumsum()
    ax2.plot(x, cum.values, color=DARK["red"], lw=2.0, zorder=5, label="Cumul TC")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_cur_fmt))
    ax2.tick_params(colors=DARK["muted"], labelsize=FS["tick"])
    ax2.set_facecolor("none")
    for sp in ax2.spines.values():
        sp.set_color(DARK["border"])

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_cur_fmt))
    ax.tick_params(axis="x", rotation=30, labelsize=FS["tick"])
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2,
              loc="upper left", framealpha=0.9, fontsize=FS["legend"])


# ─── CLASSE PRINCIPALE ────────────────────────────────────────────────────────

class PortfolioDashboardVisualizer:

    def __init__(self, ts_portfolio):
        self.ts = ts_portfolio
        _apply_dark_style()

    def _get_capital(self):
        return self.ts.metrics.get(
            "initial_capital",
            getattr(self.ts, "capital_init", 100_000)
        )

    def _get_portfolio_value(self):
        pv = self.ts.metrics.get("portfolio_value")
        if pv is not None and len(pv) > 0:
            return pv
        pnl = self.ts.metrics.get("portfolio_pnl")
        if pnl is not None:
            return pnl.cumsum() + self._get_capital()
        return None

    def _build_metrics_dict(self):
        m       = dict(self.ts.metrics)
        capital = self._get_capital()
        pf_ret  = m.get("portfolio_returns")
        pf_cum  = m.get("pf_cumulative_return")
        pnl     = m.get("portfolio_pnl")
        pv      = self._get_portfolio_value()

        if "annual_return" not in m:
            mr = m.get("mean_return")
            if mr is not None:
                m["annual_return"] = float(mr) * 252

        if "total_return" not in m and pf_cum is not None and len(pf_cum):
            m["total_return"] = float(pf_cum.iloc[-1])

        if "final_value" not in m and pv is not None and len(pv):
            m["final_value"] = float(pv.iloc[-1])

        if "total_pnl" not in m and pnl is not None:
            m["total_pnl"] = float(pnl.sum())

        if "max_drawdown" not in m:
            dd = m.get("drawdown")
            if dd is not None and len(dd):
                m["max_drawdown"] = float(dd.min())

        if "calmar_ratio" not in m:
            ar  = m.get("annual_return", 0) or 0
            mdd = m.get("max_drawdown",  0) or 0
            m["calmar_ratio"] = -ar / mdd if mdd < 0 else 0.0

        if "win_rate" not in m and pnl is not None:
            m["win_rate"] = float((pnl > 0).mean())

        if pf_ret is not None and len(pf_ret) > 10:
            r = pf_ret.dropna()
            if "var_95" not in m:
                m["var_95"] = float(np.percentile(r, 5))
            if "cvar_95" not in m:
                v95 = m["var_95"]
                m["cvar_95"] = float(r[r <= v95].mean())

        tc = m.get("transaction_costs")
        if tc is not None:
            m["tc_total"] = float(tc.sum())

        return m

    def plot_dashboard(self, figsize=(24, 36), save_path=None, dpi=150):
        """
        Layout 9 lignes :
        ┌──────────────────────────────────┐  Header
        ├──────────────────────────────────┤  KPIs
        ├─────────────────┬────────────────┤  Cum Return | Drawdown
        ├─────────────────┬────────────────┤  Rolling Sharpe | Distribution
        ├──────────────────────────────────┤  Monthly P&L
        ├──────────────────────────────────┤  Heatmap
        ├─────────────────┬────────────────┤  PnL by Type | Country Exposure
        ├─────────────────┬────────────────┤  Composition | Avg Weight
        ├──────────────────────────────────┤  TC Breakdown
        └──────────────────────────────────┘
        """
        fig = plt.figure(figsize=figsize, facecolor=DARK["bg"])
        fig.patch.set_facecolor(DARK["bg"])

        gs = GridSpec(
            9, 2, figure=fig,
            height_ratios=[0.06, 0.07, 0.13, 0.12, 0.10, 0.13, 0.12, 0.14, 0.10],
            hspace=0.65, wspace=0.28,
            left=0.06, right=0.97, top=0.965, bottom=0.025
        )

        m       = self._build_metrics_dict()
        capital = self._get_capital()

        _draw_header(fig, m, capital)

        # KPI
        ax_kpi = fig.add_subplot(gs[1, :])
        _draw_kpi_panel(ax_kpi, m)

        # Cum Return | Drawdown
        pf_cum = self.ts.metrics.get("pf_cumulative_return")
        bm_cum = self.ts.metrics.get("bm_cumulative_return")
        dd     = self.ts.metrics.get("drawdown")
        if pf_cum is not None and len(pf_cum):
            _plot_cumulative_return(fig.add_subplot(gs[2, 0]), pf_cum, bm_cum)
        if dd is not None and len(dd):
            _plot_drawdown(fig.add_subplot(gs[2, 1]), dd, m.get("max_drawdown"))

        # Rolling Sharpe | Distribution
        pf_ret = self.ts.metrics.get("portfolio_returns")
        if pf_ret is not None and len(pf_ret) > 50:
            _plot_rolling_sharpe(fig.add_subplot(gs[3, 0]), pf_ret)
            try:
                _plot_return_distribution(fig.add_subplot(gs[3, 1]), pf_ret)
            except ImportError:
                # scipy non disponible
                ax_rd = fig.add_subplot(gs[3, 1])
                _style_ax(ax_rd)
                r = pf_ret.dropna() * 100
                n_bins = min(60, max(20, len(r)//15))
                ax_rd.hist(r, bins=n_bins, color=DARK["accent"],
                           alpha=0.65, edgecolor=DARK["surface"])
                _section_title(ax_rd, "Return Distribution")

        # Monthly P&L
        pv = self._get_portfolio_value()
        if pv is not None and len(pv):
            _plot_monthly_pnl(fig.add_subplot(gs[4, :]), pv, capital)

        # Heatmap
        if pf_ret is not None and len(pf_ret):
            _plot_monthly_heatmap(fig.add_subplot(gs[5, :]), pf_ret)

        # PnL by Type | Country Exposure
        _plot_pnl_by_type(fig.add_subplot(gs[6, 0]), self.ts)
        _plot_country_exposure(fig.add_subplot(gs[6, 1]), self.ts)

        # Composition | Avg Weight
        wh = getattr(self.ts, "weights_history", pd.DataFrame())
        if not wh.empty:
            _plot_weights_composition(fig.add_subplot(gs[7, 0]), wh)
            ax_rc = fig.add_subplot(gs[7, 1])
            _style_ax(ax_rc)
            _section_title(ax_rc, "Avg Weight by Product")
            avg = wh.abs().mean().sort_values(ascending=True)
            if len(avg) > 15:
                avg = avg.nlargest(15).sort_values(ascending=True)
            total = avg.sum()
            pct   = avg / total * 100 if total > 0 else avg * 0
            colors = [ASSET_COLORS[i % len(ASSET_COLORS)] for i in range(len(pct))]
            bars  = ax_rc.barh(range(len(pct)), pct.values, color=colors,
                               alpha=0.85, edgecolor=DARK["surface"],
                               linewidth=0.4, height=0.65)
            ax_rc.set_yticks(range(len(pct)))
            ax_rc.set_yticklabels(pct.index, fontsize=FS["tick"])
            ax_rc.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
            ax_rc.grid(True, axis="x", color=DARK["grid"], linewidth=0.5)
            ax_rc.grid(False, axis="y")
            for bar, v in zip(bars, pct.values):
                ax_rc.text(v + 0.2, bar.get_y() + bar.get_height() / 2,
                           f"{v:.1f}%", va="center",
                           fontsize=FS["annotation"] - 1, color=DARK["muted"])

        # TC Breakdown
        _plot_tc_breakdown(fig.add_subplot(gs[8, :]), m)
        

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                        facecolor=DARK["bg"])
            print(f"✓ Dashboard saved → {save_path}")

        return fig

    def save_dashboard(self, filename="portfolio_dashboard.png", dpi=200):
        fig = self.plot_dashboard(save_path=filename, dpi=dpi)
        plt.close(fig)