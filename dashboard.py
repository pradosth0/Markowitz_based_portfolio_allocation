"""
PortfolioDashboardVisualizer — Bloomberg Dark Terminal Style
============================================================
Version corrigée et améliorée.

Corrections appliquées :
  - KPI : clés alignées sur TimeSeriesPortfolio.metrics (sharpe, mean_return, drawdown…)
  - plot_monthly_heatmap : utilise portfolio_returns (returns journaliers) pas pnl/capital
  - plot_drawdown : lit metrics["drawdown"] directement, ne recalcule pas
  - plot_pnl_attribution : remplacée par TC Breakdown (Markowitz vs Roll-down) — plus fictif
  - _build_metrics_dict : calcule les dérivés manquants (annual_return, calmar, var_95…)
  - Lisibilité : KPI plus grands, heatmap plus haute, moins d'annotations encombrantes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from datetime import datetime
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
    "#1A7FFF", "#00D49A", "#F0A500", "#FF3D5A",
    "#7C5CFC", "#00C4D4", "#F87C00", "#22C97A",
    "#E040FB", "#546E8A", "#A3D900", "#FF7043",
]

RYGCMAP = LinearSegmentedColormap.from_list(
    "ryg", ["#FF3D5A", "#111A2C", "#00D49A"], N=256
)


# ─── STYLE GLOBAL ─────────────────────────────────────────────────────────────

def _apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  DARK["bg"],
        "axes.facecolor":    DARK["surface"],
        "axes.edgecolor":    DARK["border"],
        "axes.linewidth":    0.7,
        "axes.labelcolor":   DARK["muted"],
        "axes.labelsize":    8,
        "axes.titlesize":    9,
        "axes.titlecolor":   DARK["text"],
        "xtick.color":       DARK["muted"],
        "ytick.color":       DARK["muted"],
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "xtick.major.size":  0,
        "ytick.major.size":  0,
        "grid.color":        DARK["grid"],
        "grid.linewidth":    0.5,
        "grid.linestyle":    "--",
        "grid.alpha":        1.0,
        "legend.facecolor":  DARK["surface"],
        "legend.edgecolor":  DARK["border"],
        "legend.labelcolor": DARK["muted"],
        "legend.fontsize":   10,
        "font.family":       "monospace",
        "text.color":        DARK["text"],
    })


# ─── FORMATTERS ───────────────────────────────────────────────────────────────

def _fmt_pct(v, decimals=1):
    return f"{v*100:+.{decimals}f}%" if not (v is None or np.isnan(v)) else "—"

def _fmt_pct_plain(v, decimals=1):
    return f"{v*100:.{decimals}f}%" if not (v is None or np.isnan(v)) else "—"

def _fmt_cur(v):
    if abs(v) >= 1e6: return f"{v/1e6:.2f}M€"
    if abs(v) >= 1e3: return f"{v/1e3:.0f}K€"
    return f"{v:.0f}€"

def _pct_formatter(v, _=None):   return f"{v*100:.1f}%"
def _cur_formatter(v, _=None):   return _fmt_cur(v)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _style_ax(ax):
    ax.set_facecolor(DARK["surface"])
    ax.tick_params(colors=DARK["muted"], which="both")
    for sp in ax.spines.values():
        sp.set_color(DARK["border"])
    ax.grid(True, color=DARK["grid"], linewidth=0.5, linestyle="--", alpha=0.9)


def _section_title(ax, title, badge=None):
    """Barre verticale colorée + titre en haut de l'axe."""
    full = title.upper() + (f"   ·  {badge}" if badge else "")
    ax.text(-0.01, 1.04, "▌", transform=ax.transAxes, fontsize=18,
            color=DARK["accent"], va="bottom", ha="right", clip_on=False, fontweight="bold")
    ax.text(0.005, 1.04, full, transform=ax.transAxes, fontsize=12,
            color=DARK["text"], va="bottom", ha="left", clip_on=False,
            fontweight="bold", fontfamily="monospace")


def _color_signed(v):
    return DARK["green"] if v >= 0 else DARK["red"]


# ─── KPI PANEL ────────────────────────────────────────────────────────────────

def _draw_kpi_panel(ax, metrics: dict):
    """
    Bande KPI — clés issues directement de TimeSeriesPortfolio.metrics
    + dérivés calculés dans _build_metrics_dict.
    """
    ax.set_facecolor(DARK["dim"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    kpis = [
        ("ANN. RETURN",
         _fmt_pct(metrics.get("annual_return")),
         _color_signed(metrics.get("annual_return", 0))),

        ("VOLATILITY",
         _fmt_pct_plain(metrics.get("volatility")),
         DARK["text"]),

        ("SHARPE",
         f"{metrics.get('sharpe', 0):.2f}",                     # ← clé corrigée
         DARK["green"] if metrics.get("sharpe", 0) > 1 else DARK["gold"]),

        ("MAX DD",
         _fmt_pct(metrics.get("max_drawdown")),
         DARK["red"]),

        ("CALMAR",
         f"{metrics.get('calmar_ratio', 0):.2f}",
         DARK["green"] if metrics.get("calmar_ratio", 0) > 1 else DARK["gold"]),

        ("WIN RATE",
         _fmt_pct_plain(metrics.get("win_rate", 0)),
         DARK["text"]),

        ("VaR 95%",
         _fmt_pct(metrics.get("var_95")),
         DARK["red"]),

        ("TOTAL P&L",
         _fmt_cur(metrics.get("total_pnl", 0)),
         _color_signed(metrics.get("total_pnl", 0))),

        ("TC TOTAL",
         _fmt_cur(metrics.get("tc_total", 0)),
         DARK["gold"]),
    ]

    n = len(kpis)
    xs = np.linspace(0, 1, n + 1)

    for i, (label, value, color) in enumerate(kpis):
        cx = (xs[i] + xs[i + 1]) / 2
        # Label
        ax.text(cx, 0.78, label, ha="center", va="center", fontsize=9.5,
                color=DARK["muted"], fontfamily="monospace", transform=ax.transAxes)
        # Valeur — plus grande
        ax.text(cx, 0.30, value, ha="center", va="center", fontsize=15,
                color=color, fontfamily="monospace", fontweight="bold",
                transform=ax.transAxes)
        # Séparateur vertical
        if i < n - 1:
            ax.axvline(xs[i + 1], color=DARK["border"], linewidth=0.8, alpha=0.6)


# ─── HEADER ───────────────────────────────────────────────────────────────────

def _draw_header(fig, metrics, capital):
    final_val  = metrics.get("final_value", capital)
    total_ret  = metrics.get("total_return", 0) or 0
    date_str   = datetime.now().strftime("%Y-%m-%d  %H:%M")

    fig.text(0.012, 0.984, "▌", fontsize=32, color=DARK["accent"], va="top")
    fig.text(0.024, 0.985, "PORTFOLIO ANALYTICS", fontsize=18,
             color=DARK["text"], fontweight="bold", fontfamily="monospace", va="top")
    fig.text(0.024, 0.974, "Fixed Income · Multi-Product · DV01-Normalized",
             fontsize=12, color=DARK["muted"], fontfamily="monospace", va="top")

    fig.text(0.60, 0.983, "NAV", fontsize=9.5, color=DARK["muted"],
             fontfamily="monospace", va="top")
    fig.text(0.60, 0.973, _fmt_cur(final_val), fontsize=15,
             color=DARK["green"], fontweight="bold", fontfamily="monospace", va="top")

    fig.text(0.72, 0.983, "TOTAL RETURN", fontsize=9.5, color=DARK["muted"],
             fontfamily="monospace", va="top")
    fig.text(0.72, 0.973, _fmt_pct(total_ret), fontsize=15,
             color=_color_signed(total_ret), fontweight="bold",
             fontfamily="monospace", va="top")

    fig.text(0.98, 0.983, date_str, fontsize=9.5, color=DARK["muted"],
             fontfamily="monospace", va="top", ha="right")

    fig.add_artist(plt.Line2D(
        [0.01, 0.99], [0.966, 0.966], transform=fig.transFigure,
        color=DARK["border"], linewidth=0.8
    ))


# ─── 1. CUMULATIVE RETURN ─────────────────────────────────────────────────────

def _plot_cumulative_return(ax, pf_cum, bm_cum=None):
    _style_ax(ax)
    badge = _fmt_pct(float(pf_cum.iloc[-1])) if len(pf_cum) else ""
    _section_title(ax, "Cumulative Return", badge)

    x = pf_cum.index
    ax.plot(x, pf_cum.values, color=DARK["accent"], lw=2, label="Portfolio", zorder=4)
    ax.fill_between(x, pf_cum.values, 0, color=DARK["accent"], alpha=0.08)

    if bm_cum is not None and len(bm_cum):
        ax.plot(x, bm_cum.values, color=DARK["gold"], lw=1.4,
                linestyle="--", alpha=0.8, label="Benchmark", zorder=3)
        ax.fill_between(x, pf_cum.values, bm_cum.values,
                        where=pf_cum.values >= bm_cum.values,
                        color=DARK["green"], alpha=0.12, interpolate=True)
        ax.fill_between(x, pf_cum.values, bm_cum.values,
                        where=pf_cum.values < bm_cum.values,
                        color=DARK["red"], alpha=0.12, interpolate=True)
        # Annotation benchmark finale
        ax.annotate(_fmt_pct(float(bm_cum.iloc[-1])),
                    xy=(bm_cum.index[-1], bm_cum.iloc[-1]),
                    xytext=(-55, -14), textcoords="offset points",
                    fontsize=10, color=DARK["gold"],
                    arrowprops=dict(arrowstyle="-", color=DARK["muted"], lw=0.6))

    ax.axhline(0, color=DARK["border"], lw=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_pct_formatter))
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="upper left", framealpha=0.85)

    # Annotation portfolio finale
    ax.annotate(_fmt_pct(float(pf_cum.iloc[-1])),
                xy=(pf_cum.index[-1], pf_cum.iloc[-1]),
                xytext=(-55, 10), textcoords="offset points",
                fontsize=10.5, color=DARK["accent"], fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=DARK["muted"], lw=0.6))


# ─── 2. DRAWDOWN ──────────────────────────────────────────────────────────────

def _plot_drawdown(ax, dd_series, max_dd=None):
    """
    ✅ Corrigé : reçoit directement metrics["drawdown"] — ne recalcule plus.
    """
    _style_ax(ax)
    badge = f"Max {_fmt_pct(max_dd)}" if max_dd is not None else ""
    _section_title(ax, "Drawdown", badge)

    x = dd_series.index
    ax.fill_between(x, dd_series.values, 0, color=DARK["red"], alpha=0.35, zorder=2)
    ax.plot(x, dd_series.values, color=DARK["red"], lw=1.2, zorder=3)
    ax.axhline(0, color=DARK["border"], lw=0.8)

    # Annotation du max drawdown
    if max_dd is not None and not np.isnan(max_dd):
        idx_min = dd_series.idxmin()
        ax.annotate(_fmt_pct(max_dd),
                    xy=(idx_min, max_dd),
                    xytext=(30, -18), textcoords="offset points",
                    fontsize=10, color=DARK["red"], fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=DARK["muted"], lw=0.6))

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    ax.tick_params(axis="x", rotation=25)


# ─── 3. MONTHLY P&L WATERFALL ─────────────────────────────────────────────────

def _plot_monthly_pnl(ax, portfolio_value, capital):
    _style_ax(ax)
    monthly   = portfolio_value.resample("ME").last().ffill()
    cum_pnl   = monthly - capital
    month_ret = cum_pnl.diff().fillna(cum_pnl.iloc[0])
    bottoms   = cum_pnl.shift(1).fillna(0)

    _section_title(ax, "Monthly P&L  (Cumulative)", _fmt_cur(float(cum_pnl.iloc[-1])))

    colors = [DARK["green"] if v >= 0 else DARK["red"] for v in month_ret]
    ax.bar(monthly.index, month_ret.values, bottom=bottoms.values,
           color=colors, alpha=0.80, width=20,
           edgecolor=DARK["surface"], linewidth=0.4)
    ax.plot(cum_pnl.index, cum_pnl.values, color=DARK["text"],
            lw=1.5, marker="o", markersize=2.5, zorder=4)
    ax.axhline(0, color=DARK["border"], lw=0.8)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_cur_formatter))
    ax.tick_params(axis="x", rotation=35)


# ─── 4. MONTHLY HEATMAP ───────────────────────────────────────────────────────

def _plot_monthly_heatmap(ax, portfolio_returns):
    """
    ✅ Corrigé : utilise portfolio_returns (Series de returns journaliers)
    au lieu de pnl/capital — évite les doublons et les erreurs d'échelle.
    """
    _style_ax(ax)
    _section_title(ax, "Monthly Returns Heatmap")

    # Returns mensuels composés
    monthly = (1 + portfolio_returns.fillna(0)).resample("ME").prod() - 1
    monthly.index = pd.to_datetime(monthly.index)

    years  = sorted(monthly.index.year.unique())
    months = range(1, 13)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    mat = np.full((len(years), 12), np.nan)
    for i, y in enumerate(years):
        for j, m in enumerate(months):
            mask = (monthly.index.year == y) & (monthly.index.month == m)
            if mask.any():
                mat[i, j] = monthly[mask].values[0] * 100

    vmax = np.nanpercentile(np.abs(mat), 90) or 1.0
    im = ax.imshow(mat, aspect="auto", cmap=RYGCMAP,
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels, fontsize=10.5, color=DARK["muted"])
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels([str(y) for y in years], fontsize=10.5, color=DARK["muted"])
    ax.tick_params(length=0)
    ax.grid(False)

    # Annotations lisibles — taille fixe, couleur adaptée à l'intensité
    for i in range(len(years)):
        for j in range(12):
            v = mat[i, j]
            if not np.isnan(v):
                txt_col = DARK["text"] if abs(v) < vmax * 0.6 else DARK["bg"]
                ax.text(j, i, f"{v:+.1f}%", ha="center", va="center",
                        fontsize=10, color=txt_col, fontfamily="monospace",
                        fontweight="bold" if abs(v) > vmax * 0.5 else "normal")

    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.01, shrink=0.9)
    cbar.ax.tick_params(colors=DARK["muted"], labelsize=7.5)
    cbar.outline.set_edgecolor(DARK["border"])
    cbar.set_label("Return (%)", color=DARK["muted"], fontsize=9.5)


# ─── 5. WEIGHTS COMPOSITION ───────────────────────────────────────────────────

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
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="upper left", ncol=2, fontsize=9.5,
              framealpha=0.9, labelcolor=DARK["text"])


# ─── 6. RISK CONTRIBUTION ─────────────────────────────────────────────────────

def _plot_risk_contribution(ax, weights_history):
    _style_ax(ax)
    _section_title(ax, "Avg Weight by Product")

    avg    = weights_history.abs().mean().sort_values(ascending=True)
    total  = avg.sum()
    pct    = (avg / total * 100) if total > 0 else avg * 0

    # Limiter à top 15 pour la lisibilité
    if len(pct) > 15:
        pct = pct.nlargest(15).sort_values(ascending=True)

    colors = [ASSET_COLORS[i % len(ASSET_COLORS)] for i in range(len(pct))]
    bars   = ax.barh(range(len(pct)), pct.values,
                     color=colors, alpha=0.85,
                     edgecolor=DARK["surface"], linewidth=0.4, height=0.65)

    ax.set_yticks(range(len(pct)))
    ax.set_yticklabels(pct.index, fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(True, axis="x", color=DARK["grid"], linewidth=0.5)
    ax.grid(False, axis="y")

    for bar, v in zip(bars, pct.values):
        ax.text(v + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=9.5,
                color=DARK["muted"], fontfamily="monospace")


# ─── 7. TRANSACTION COSTS BREAKDOWN ──────────────────────────────────────────

def _plot_tc_breakdown(ax, metrics):
    """
    ✅ Remplace plot_pnl_attribution (qui utilisait np.random).
    Affiche TC Markowitz vs TC Roll-down par période de rebalancement.
    """
    _style_ax(ax)

    tc_mkt  = metrics.get("transaction_costs_markowitz")
    tc_roll = metrics.get("transaction_costs_rolldown")
    tc_tot  = metrics.get("transaction_costs")

    # Fallback si les séries détaillées ne sont pas disponibles
    if tc_mkt is None or tc_roll is None:
        if tc_tot is not None:
            _section_title(ax, "Transaction Costs", _fmt_cur(float(tc_tot.sum())))
            ax.bar(tc_tot.index, tc_tot.values,
                   color=DARK["gold"], alpha=0.8, width=20,
                   edgecolor=DARK["surface"], linewidth=0.4)
            ax.plot(tc_tot.index, tc_tot.cumsum(),
                    color=DARK["red"], lw=1.5, label="Cumulative TC", zorder=4)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(_cur_formatter))
            ax.tick_params(axis="x", rotation=30)
            ax.legend(loc="upper left", framealpha=0.85)
        else:
            ax.text(0.5, 0.5, "No TC data available", ha="center", va="center",
                    transform=ax.transAxes, color=DARK["muted"], fontsize=12)
        return

    total_tc = float(tc_tot.sum()) if tc_tot is not None else float(tc_mkt.sum() + tc_roll.sum())
    _section_title(ax, "Transaction Costs — Markowitz vs Roll-Down",
                   f"Total {_fmt_cur(total_tc)}")

    x     = tc_mkt.index
    w     = (x[1] - x[0]).days * 0.4 if len(x) > 1 else 15
    w     = max(w, 10)

    ax.bar(x, tc_mkt.values,  width=w, label="Markowitz rebalance",
           color=DARK["accent"], alpha=0.82, edgecolor=DARK["surface"], linewidth=0.4)
    ax.bar(x, tc_roll.values, width=w, bottom=tc_mkt.values, label="Roll-down",
           color=DARK["gold"], alpha=0.82, edgecolor=DARK["surface"], linewidth=0.4)

    # Cumul en ligne secondaire
    ax2 = ax.twinx()
    cum = (tc_mkt + tc_roll).cumsum()
    ax2.plot(x, cum.values, color=DARK["red"], lw=1.8,
             linestyle="-", zorder=5, label="Cumul TC")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_cur_formatter))
    ax2.tick_params(colors=DARK["muted"])
    ax2.set_facecolor("none")
    for sp in ax2.spines.values():
        sp.set_color(DARK["border"])

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_cur_formatter))
    ax.tick_params(axis="x", rotation=30)

    # Légende combinée
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              loc="upper left", framealpha=0.9, fontsize=10)


# ─── CLASSE PRINCIPALE ────────────────────────────────────────────────────────

class PortfolioDashboardVisualizer:
    """
    Bloomberg Dark Terminal — Dashboard complet.

    Compatible avec TimeSeriesPortfolio.metrics :
        pf_cumulative_return, bm_cumulative_return,
        portfolio_pnl, portfolio_returns, portfolio_value,
        sharpe, mean_return, volatility, drawdown, max_drawdown,
        transaction_costs, transaction_costs_markowitz, transaction_costs_rolldown
    """

    def __init__(self, ts_portfolio):
        self.ts = ts_portfolio
        _apply_dark_style()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_capital(self):
        return self.ts.metrics.get(
            "initial_capital",
            getattr(self.ts, "capital_init", 1_000_000)
        )

    def _get_portfolio_value(self):
        pv = self.ts.metrics.get("portfolio_value")
        if pv is not None and len(pv) > 0:
            return pv
        # Fallback : reconstituer depuis pnl
        pnl = self.ts.metrics.get("portfolio_pnl")
        if pnl is not None:
            return pnl.cumsum() + self._get_capital()
        return None

    def _build_metrics_dict(self):
        """
        ✅ Corrigé : calcule les métriques dérivées manquantes depuis les
        clés natives de TimeSeriesPortfolio (sharpe, mean_return, drawdown…).
        """
        m       = dict(self.ts.metrics)
        capital = self._get_capital()

        pf_ret = m.get("portfolio_returns", m.get("pf_cumulative_return"))
        pf_cum = m.get("pf_cumulative_return")
        pnl    = m.get("portfolio_pnl")
        pv     = self._get_portfolio_value()

        # Annual return (depuis mean_return journalier × 252)
        if "annual_return" not in m:
            mr = m.get("mean_return")
            if mr is not None:
                m["annual_return"] = float(mr) * 252

        # Total return
        if "total_return" not in m and pf_cum is not None and len(pf_cum):
            m["total_return"] = float(pf_cum.iloc[-1])

        # Final value
        if "final_value" not in m and pv is not None and len(pv):
            m["final_value"] = float(pv.iloc[-1])

        # Total PnL
        if "total_pnl" not in m and pnl is not None:
            m["total_pnl"] = float(pnl.sum())

        # Max drawdown (depuis la série drawdown)
        if "max_drawdown" not in m:
            dd = m.get("drawdown")
            if dd is not None and len(dd):
                m["max_drawdown"] = float(dd.min())

        # Calmar ratio
        if "calmar_ratio" not in m:
            ar  = m.get("annual_return", 0) or 0
            mdd = m.get("max_drawdown",  0) or 0
            m["calmar_ratio"] = -ar / mdd if mdd < 0 else 0.0

        # Win rate
        if "win_rate" not in m and pnl is not None:
            m["win_rate"] = float((pnl > 0).mean())

        # VaR 95% journalier
        if "var_95" not in m:
            ret_series = m.get("portfolio_returns")
            if ret_series is not None and len(ret_series) > 10:
                m["var_95"] = float(np.percentile(ret_series.dropna(), 5))

        # TC total
        tc = m.get("transaction_costs")
        if tc is not None:
            m["tc_total"] = float(tc.sum())

        return m

    # ── Dashboard principal ───────────────────────────────────────────────────

    def plot_dashboard(self, figsize=(22, 28), save_path=None, dpi=150):
        """
        Layout 7 lignes :
        ┌─────────────────────────────────┐  Header
        ├─────────────────────────────────┤  KPIs (haute)
        ├──────────────────┬──────────────┤  Cum Return  │ Drawdown
        ├─────────────────────────────────┤  Monthly P&L
        ├─────────────────────────────────┤  Heatmap (haute)
        ├──────────────────┬──────────────┤  Composition │ Risk Contrib
        ├─────────────────────────────────┤  TC Breakdown
        └─────────────────────────────────┘
        """
        fig = plt.figure(figsize=figsize, facecolor=DARK["bg"])
        fig.patch.set_facecolor(DARK["bg"])

        gs = GridSpec(
            7, 2, figure=fig,
            height_ratios=[0.07, 0.09, 0.18, 0.14, 0.17, 0.19, 0.14],
            hspace=0.58, wspace=0.28,
            left=0.07, right=0.97, top=0.965, bottom=0.03
        )

        m       = self._build_metrics_dict()
        capital = self._get_capital()

        # ── Header ──────────────────────────────────────────────────────────
        _draw_header(fig, m, capital)

        # ── KPI ─────────────────────────────────────────────────────────────
        ax_kpi = fig.add_subplot(gs[1, :])
        _draw_kpi_panel(ax_kpi, m)

        # ── Cum Return │ Drawdown ────────────────────────────────────────────
        pf_cum = self.ts.metrics.get("pf_cumulative_return")
        bm_cum = self.ts.metrics.get("bm_cumulative_return")
        dd     = self.ts.metrics.get("drawdown")

        if pf_cum is not None and len(pf_cum):
            ax_cum = fig.add_subplot(gs[2, 0])
            _plot_cumulative_return(ax_cum, pf_cum, bm_cum)

        if dd is not None and len(dd):
            ax_dd = fig.add_subplot(gs[2, 1])
            _plot_drawdown(ax_dd, dd, m.get("max_drawdown"))

        # ── Monthly P&L ──────────────────────────────────────────────────────
        pv = self._get_portfolio_value()
        if pv is not None and len(pv):
            ax_pnl = fig.add_subplot(gs[3, :])
            _plot_monthly_pnl(ax_pnl, pv, capital)

        # ── Heatmap ──────────────────────────────────────────────────────────
        pf_returns = self.ts.metrics.get("portfolio_returns")
        if pf_returns is not None and len(pf_returns):
            ax_hm = fig.add_subplot(gs[4, :])
            _plot_monthly_heatmap(ax_hm, pf_returns)

        # ── Composition │ Risk Contrib ───────────────────────────────────────
        wh = getattr(self.ts, "weights_history", pd.DataFrame())
        if not wh.empty:
            ax_w  = fig.add_subplot(gs[5, 0])
            _plot_weights_composition(ax_w, wh)

            ax_rc = fig.add_subplot(gs[5, 1])
            _plot_risk_contribution(ax_rc, wh)

        # ── TC Breakdown ─────────────────────────────────────────────────────
        ax_tc = fig.add_subplot(gs[6, :])
        _plot_tc_breakdown(ax_tc, m)

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
                        facecolor=DARK["bg"])
            print(f"✓ Dashboard saved → {save_path}")

        return fig

    def save_dashboard(self, filename="portfolio_dashboard.png", dpi=200):
        fig = self.plot_dashboard(save_path=filename, dpi=dpi)
        plt.close(fig)


# ─── DEMO ─────────────────────────────────────────────────────────────────────

class _MockPortfolio:
    """Portfolio simulé pour démonstration standalone."""

    def __init__(self):
        self.capital_init = 1_000_000
        dates = pd.bdate_range("2021-01-04", "2025-06-10")
        np.random.seed(42)

        pf_ret  = pd.Series(np.random.normal(0.00045, 0.008, len(dates)), index=dates)
        bm_ret  = pd.Series(np.random.normal(0.00020, 0.006, len(dates)), index=dates)
        pf_pnl  = pf_ret * self.capital_init
        pf_cum  = (1 + pf_ret).cumprod() - 1
        bm_cum  = (1 + bm_ret).cumprod() - 1
        pf_val  = self.capital_init * (1 + pf_cum)
        roll_mx = pf_val.cummax()
        dd      = (pf_val - roll_mx) / roll_mx
        ann_ret = float((1 + pf_cum.iloc[-1]) ** (252 / len(dates)) - 1)
        vol     = float(pf_ret.std() * np.sqrt(252))

        # TC simulées
        rb_dates = pd.bdate_range("2021-01-04", "2025-06-10", freq="ME")
        tc_mkt   = pd.Series(np.abs(np.random.normal(500, 200, len(rb_dates))), index=rb_dates)
        tc_roll  = pd.Series(np.abs(np.random.normal(150, 80,  len(rb_dates))), index=rb_dates)

        self.metrics = {
            "pf_cumulative_return":        pf_cum,
            "bm_cumulative_return":        bm_cum,
            "portfolio_returns":           pf_ret,
            "portfolio_pnl":               pf_pnl,
            "portfolio_value":             pf_val,
            "initial_capital":             self.capital_init,
            "mean_return":                 float(pf_ret.mean()),
            "volatility":                  vol,
            "sharpe":                      ann_ret / vol if vol else 0,
            "drawdown":                    dd,
            "max_drawdown":                float(dd.min()),
            "win_rate":                    float((pf_ret > 0).mean()),
            "var_95":                      float(np.percentile(pf_ret, 5)),
            "total_pnl":                   float(pf_pnl.sum()),
            "total_return":                float(pf_cum.iloc[-1]),
            "final_value":                 float(pf_val.iloc[-1]),
            "transaction_costs":           tc_mkt + tc_roll,
            "transaction_costs_markowitz": tc_mkt,
            "transaction_costs_rolldown":  tc_roll,
        }

        assets   = ["FR_2","DE_5","IT_10","FR_DE_5","FR_2_10",
                    "DE_fly_2_5_10","FR_DE_slope_2_10","SP_10","DE_2"]
        rb_idx   = pd.bdate_range("2021-01-04", "2025-06-10", freq="ME")
        wh       = pd.DataFrame(
            np.random.dirichlet(np.ones(len(assets)), len(rb_idx)),
            index=rb_idx, columns=assets
        )
        self.weights_history = wh.reindex(dates).ffill().fillna(0)
        self.rebalance_dates = rb_idx



if __name__ == "__main__":
    print("Generating dashboard...")
    mock = _MockPortfolio()
    viz  = PortfolioDashboardVisualizer(mock)
    fig  = viz.plot_dashboard(save_path="portfolio_dashboard.png", dpi=150)
    plt.show()
    print("Done.")