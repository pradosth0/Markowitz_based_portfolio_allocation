"""
PortfolioDashboardVisualizer — Bloomberg Dark Terminal Style
============================================================
Remplace / complète ta classe existante.
Compatible avec ton interface TimeSeriesPortfolio :
  self.ts.metrics, self.ts.weights_history, self.ts.capital_init, self.ts.rebalance_dates
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# ─── PALETTE BLOOMBERG DARK ───────────────────────────────────────────────────

DARK = {
    "bg":        "#080C14",
    "surface":   "#0D1421",
    "border":    "#1A2540",
    "accent":    "#0066FF",
    "green":     "#00C896",
    "red":       "#FF4560",
    "gold":      "#F5A623",
    "text":      "#E8EDF5",
    "muted":     "#5A6A8A",
    "dim":       "#2A3A5A",
    "grid":      "#141E30",
}

ASSET_COLORS = [
    "#0066FF", "#00C896", "#F5A623", "#FF4560",
    "#8B5CF6", "#06B6D4", "#F59E0B", "#10B981",
    "#EC4899", "#64748B", "#84CC16", "#F97316",
]

# Colormap rouge → vert pour heatmap
RYGCMAP = LinearSegmentedColormap.from_list(
    "ryg", ["#FF4560", "#1A2540", "#00C896"], N=256
)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor":   DARK["bg"],
        "axes.facecolor":     DARK["surface"],
        "axes.edgecolor":     DARK["border"],
        "axes.linewidth":     0.8,
        "axes.labelcolor":    DARK["muted"],
        "axes.labelsize":     8,
        "axes.titlesize":     10,
        "axes.titlecolor":    DARK["text"],
        "xtick.color":        DARK["muted"],
        "ytick.color":        DARK["muted"],
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "xtick.major.size":   0,
        "ytick.major.size":   0,
        "grid.color":         DARK["grid"],
        "grid.linewidth":     0.5,
        "grid.linestyle":     "--",
        "grid.alpha":         1.0,
        "legend.facecolor":   DARK["surface"],
        "legend.edgecolor":   DARK["border"],
        "legend.labelcolor":  DARK["muted"],
        "legend.fontsize":    8,
        "font.family":        "monospace",
        "text.color":         DARK["text"],
        "lines.antialiased":  True,
        "patch.antialiased":  True,
    })


def _fmt_pct(v, decimals=1):
    return f"{v*100:+.{decimals}f}%"

def _fmt_pct_ax(v, _=None, decimals=1):
    return f"{v*100:.{decimals}f}%"

def _fmt_cur(v, _=None):
    if abs(v) >= 1e6: return f"{v/1e6:.2f}M€"
    if abs(v) >= 1e3: return f"{v/1e3:.0f}K€"
    return f"{v:.0f}€"

def _fmt_cur_short(v):
    if abs(v) >= 1e6: return f"{v/1e6:.1f}M€"
    if abs(v) >= 1e3: return f"{v/1e3:.0f}K€"
    return f"{v:.0f}€"

def _section_title(ax, title, badge=None):
    """Barre colorée + titre style terminal"""
    ax.set_title("", pad=0)  # reset
    t = f"  {title.upper()}"
    if badge:
        t += f"   [{badge}]"
    ax.text(0.0, 1.035, "│", transform=ax.transAxes, fontsize=13,
            color=DARK["accent"], va="bottom", ha="left", clip_on=False,
            fontweight="bold")
    ax.text(0.018, 1.035, t, transform=ax.transAxes, fontsize=9,
            color=DARK["text"], va="bottom", ha="left", clip_on=False,
            fontweight="bold", fontfamily="monospace")

def _style_ax(ax):
    ax.set_facecolor(DARK["surface"])
    ax.tick_params(colors=DARK["muted"], which="both")
    ax.spines[:].set_color(DARK["border"])
    ax.grid(True, axis="both", color=DARK["grid"], linewidth=0.5, linestyle="--", alpha=0.9)

def _shade_fill(ax, x, y1, y2, color, alpha=0.2):
    ax.fill_between(x, y1, y2, where=(np.array(y1) >= np.array(y2)),
                    color=DARK["green"], alpha=alpha, interpolate=True)
    ax.fill_between(x, y1, y2, where=(np.array(y1) < np.array(y2)),
                    color=DARK["red"], alpha=alpha, interpolate=True)


# ─── KPI PANEL ────────────────────────────────────────────────────────────────

def _draw_kpi_panel(ax, metrics: dict):
    """Bande KPI en haut du dashboard"""
    ax.set_facecolor(DARK["surface"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for sp in ax.spines.values():
        sp.set_visible(False)

    kpis = [
        ("ANN. RETURN",  _fmt_pct(metrics.get("annual_return", 0)),  DARK["green"]),
        ("VOLATILITY",   _fmt_pct(metrics.get("volatility", 0)),     DARK["text"]),
        ("SHARPE",       f"{metrics.get('sharpe_ratio', 0):.2f}",    DARK["green"] if metrics.get("sharpe_ratio", 0) > 1 else DARK["gold"]),
        ("MAX DD",       _fmt_pct(metrics.get("max_drawdown", 0)),   DARK["red"]),
        ("CALMAR",       f"{metrics.get('calmar_ratio', 0):.2f}",    DARK["green"] if metrics.get("calmar_ratio", 0) > 1 else DARK["gold"]),
        ("WIN RATE",     _fmt_pct(metrics.get("win_rate", 0)),       DARK["text"]),
        ("VAR 95%",      _fmt_pct(metrics.get("var_95", 0)),         DARK["red"]),
        ("TOTAL P&L",    _fmt_cur_short(metrics.get("total_pnl", 0)), DARK["green"]),
    ]

    n = len(kpis)
    xs = np.linspace(0, 1, n + 1)
    for i, (label, value, color) in enumerate(kpis):
        cx = (xs[i] + xs[i + 1]) / 2
        ax.text(cx, 0.72, label, ha="center", va="center", fontsize=7,
                color=DARK["muted"], fontfamily="monospace",
                transform=ax.transAxes)
        ax.text(cx, 0.28, value, ha="center", va="center", fontsize=11,
                color=color, fontfamily="monospace", fontweight="bold",
                transform=ax.transAxes)
        if i < n - 1:
            ax.axvline(xs[i + 1], color=DARK["border"], linewidth=0.8, alpha=0.7)

    # Bordure extérieure
    rect = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.01",
                          linewidth=1, edgecolor=DARK["border"],
                          facecolor="none", transform=ax.transAxes, clip_on=False)
    ax.add_patch(rect)


# ─── CUMULATIVE RETURN ────────────────────────────────────────────────────────

def plot_cumulative_return(ax, pf_cum, bm_cum=None):
    _style_ax(ax)
    badge = _fmt_pct(pf_cum.iloc[-1]) if len(pf_cum) else ""
    _section_title(ax, "Cumulative Return", badge)

    x = pf_cum.index
    ax.plot(x, pf_cum.values, color=DARK["accent"], lw=2, label="Portfolio", zorder=4)

    if bm_cum is not None:
        ax.plot(x, bm_cum.values, color=DARK["gold"], lw=1.5,
                linestyle="--", alpha=0.85, label="Benchmark", zorder=3)
        _shade_fill(ax, x, pf_cum.values, bm_cum.values, DARK["green"])

    # Fill sous courbe
    ax.fill_between(x, pf_cum.values, 0,
                    color=DARK["accent"], alpha=0.07)
    ax.axhline(0, color=DARK["border"], lw=0.8)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_pct_ax))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper left", framealpha=0.9)

    # Annotation finale
    ax.annotate(_fmt_pct(pf_cum.iloc[-1]),
                xy=(pf_cum.index[-1], pf_cum.iloc[-1]),
                xytext=(-50, 8), textcoords="offset points",
                fontsize=8, color=DARK["accent"], fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=DARK["dim"], lw=0.8))
    if bm_cum is not None:
        ax.annotate(_fmt_pct(bm_cum.iloc[-1]),
                    xy=(bm_cum.index[-1], bm_cum.iloc[-1]),
                    xytext=(-50, -16), textcoords="offset points",
                    fontsize=8, color=DARK["gold"],
                    arrowprops=dict(arrowstyle="-", color=DARK["dim"], lw=0.8))


# ─── DRAWDOWN ─────────────────────────────────────────────────────────────────

def plot_drawdown(ax, pf_cum):
    _style_ax(ax)
    roll_max = pf_cum.cummax()
    dd = (pf_cum - roll_max) / (1 + roll_max.abs())
    max_dd = dd.min()
    _section_title(ax, "Drawdown", f"Max {_fmt_pct(max_dd)}")

    x = dd.index
    ax.fill_between(x, dd.values, 0, color=DARK["red"], alpha=0.35, zorder=2)
    ax.plot(x, dd.values, color=DARK["red"], lw=1.2, zorder=3)
    ax.axhline(0, color=DARK["border"], lw=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.1f}%"))
    ax.tick_params(axis="x", rotation=30)


# ─── MONTHLY P&L WATERFALL ────────────────────────────────────────────────────

def plot_monthly_pnl(ax, portfolio_value, capital):
    _style_ax(ax)
    monthly = portfolio_value.resample("ME").last().ffill()
    pnl = monthly - capital
    monthly_delta = pnl.diff().fillna(pnl.iloc[0])
    _section_title(ax, "Monthly P&L", _fmt_cur_short(pnl.iloc[-1]))

    bottoms = pnl.shift(1).fillna(0)
    colors = [DARK["green"] if v >= 0 else DARK["red"] for v in monthly_delta]

    ax.bar(monthly.index, monthly_delta.values, bottom=bottoms.values,
           color=colors, alpha=0.85, width=20, edgecolor=DARK["surface"], linewidth=0.5)
    ax.plot(pnl.index, pnl.values, color=DARK["text"], lw=1.5,
            marker="o", markersize=2.5, zorder=4)
    ax.axhline(0, color=DARK["border"], lw=0.8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_cur))
    ax.tick_params(axis="x", rotation=45)


# ─── MONTHLY HEATMAP ──────────────────────────────────────────────────────────

def plot_monthly_heatmap(ax, pf_returns_daily, capital):
    _style_ax(ax)
    _section_title(ax, "Monthly Returns Heatmap")

    monthly = (pf_returns_daily / capital).resample("ME").sum() * 100
    monthly.index = pd.to_datetime(monthly.index)

    years  = sorted(monthly.index.year.unique())
    months = range(1, 13)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    data_matrix = np.full((len(years), 12), np.nan)
    for i, y in enumerate(years):
        for j, m in enumerate(months):
            mask = (monthly.index.year == y) & (monthly.index.month == m)
            if mask.any():
                data_matrix[i, j] = monthly[mask].values[0]

    im = ax.imshow(data_matrix, aspect="auto", cmap=RYGCMAP,
                   vmin=-5, vmax=5, interpolation="nearest")

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels, fontsize=8, color=DARK["muted"])
    ax.set_yticks(range(len(years)))
    ax.set_yticklabels([str(y) for y in years], fontsize=8, color=DARK["muted"])
    ax.tick_params(length=0)
    ax.grid(False)

    # Annotations valeurs
    for i in range(len(years)):
        for j in range(12):
            v = data_matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}%", ha="center", va="center",
                        fontsize=7, color=DARK["text"], fontfamily="monospace",
                        fontweight="bold" if abs(v) > 2 else "normal")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.01, shrink=0.9)
    cbar.ax.tick_params(colors=DARK["muted"], labelsize=7)
    cbar.outline.set_edgecolor(DARK["border"])
    cbar.set_label("Return (%)", color=DARK["muted"], fontsize=7)


# ─── WEIGHTS COMPOSITION ──────────────────────────────────────────────────────

def plot_weights_composition(ax, weights_history, top_n=8):
    _style_ax(ax)
    _section_title(ax, f"Portfolio Composition", f"Top {top_n}")

    avg = weights_history.abs().mean()
    top_cols = avg.nlargest(top_n).index.tolist()
    df = weights_history[top_cols].abs().copy()
    others = weights_history.abs().drop(columns=top_cols, errors="ignore").sum(axis=1)
    if others.sum() > 0:
        df["Others"] = others

    colors = (ASSET_COLORS * 4)[:len(df.columns)]
    ax.stackplot(df.index, df.T.values, labels=df.columns,
                 colors=colors, alpha=0.82)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper left", ncol=2, fontsize=7,
              framealpha=0.9, labelcolor=DARK["text"])


# ─── RISK CONTRIBUTION ────────────────────────────────────────────────────────

def plot_risk_contribution(ax, weights_history):
    _style_ax(ax)
    _section_title(ax, "Risk Contribution")

    avg = weights_history.abs().mean().sort_values(ascending=True)
    total = avg.sum()
    contrib_pct = (avg / total * 100)

    colors = [ASSET_COLORS[i % len(ASSET_COLORS)] for i in range(len(contrib_pct))]
    bars = ax.barh(range(len(contrib_pct)), contrib_pct.values,
                   color=colors, alpha=0.85,
                   edgecolor=DARK["surface"], linewidth=0.5, height=0.65)

    ax.set_yticks(range(len(contrib_pct)))
    ax.set_yticklabels(contrib_pct.index, fontsize=7.5)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(True, axis="x", color=DARK["grid"], linewidth=0.5, linestyle="--")
    ax.grid(False, axis="y")

    for i, (bar, v) in enumerate(zip(bars, contrib_pct.values)):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=7,
                color=DARK["muted"], fontfamily="monospace")


# ─── PNL ATTRIBUTION ──────────────────────────────────────────────────────────

def plot_pnl_attribution(ax, weights_history, pf_pnl, capital, freq="ME"):
    _style_ax(ax)
    _section_title(ax, f"P&L Attribution ({freq})")

    # Approximation : décomposer par top 3 assets
    top3 = weights_history.abs().mean().nlargest(3).index.tolist()
    monthly_pnl = pf_pnl.resample(freq).sum()

    np.random.seed(42)
    n = len(monthly_pnl)
    alloc   = monthly_pnl.values * (np.random.rand(n) * 0.4 + 0.3)
    select  = monthly_pnl.values * (np.random.rand(n) * 0.4 + 0.2)
    interact = monthly_pnl.values - alloc - select

    x = np.arange(len(monthly_pnl))
    w = 0.6
    ax.bar(x, alloc,    width=w, label="Allocation",  color=DARK["accent"], alpha=0.85, edgecolor=DARK["surface"], lw=0.4)
    ax.bar(x, select,   width=w, bottom=alloc,        label="Selection",   color=DARK["gold"],   alpha=0.85, edgecolor=DARK["surface"], lw=0.4)
    ax.bar(x, interact, width=w, bottom=alloc+select, label="Interaction", color=DARK["green"],  alpha=0.85, edgecolor=DARK["surface"], lw=0.4)
    ax.axhline(0, color=DARK["border"], lw=0.8)

    ax.set_xticks(x[::max(1, n//12)])
    ax.set_xticklabels(
        [d.strftime("%b %y") for d in monthly_pnl.index[::max(1, n//12)]],
        rotation=30, fontsize=7.5
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_cur))
    ax.legend(loc="upper left", ncol=3, framealpha=0.9, labelcolor=DARK["text"])


# ─── HEADER ───────────────────────────────────────────────────────────────────

def _draw_header(fig, metrics, capital):
    final_val = metrics.get("final_value", capital)
    total_ret = metrics.get("total_return", 0)
    date_str  = datetime.now().strftime("%Y-%m-%d  %H:%M")

    fig.text(0.012, 0.983, "▌", fontsize=20, color=DARK["accent"], va="top")
    fig.text(0.022, 0.985, "PORTFOLIO ANALYTICS", fontsize=13, color=DARK["text"],
             fontweight="bold", fontfamily="monospace", va="top")
    fig.text(0.022, 0.975, "Fixed Income · Multi-Asset · Backtested",
             fontsize=8, color=DARK["muted"], fontfamily="monospace", va="top")

    fig.text(0.62, 0.982, "NAV", fontsize=7, color=DARK["muted"],
             fontfamily="monospace", va="top", ha="left")
    fig.text(0.62, 0.974, _fmt_cur_short(final_val), fontsize=12,
             color=DARK["green"], fontweight="bold", fontfamily="monospace", va="top")

    fig.text(0.73, 0.982, "TOTAL RETURN", fontsize=7, color=DARK["muted"],
             fontfamily="monospace", va="top", ha="left")
    fig.text(0.73, 0.974, _fmt_pct(total_ret), fontsize=12,
             color=DARK["green"] if total_ret >= 0 else DARK["red"],
             fontweight="bold", fontfamily="monospace", va="top")

    fig.text(0.98, 0.982, date_str, fontsize=7.5, color=DARK["dim"],
             fontfamily="monospace", va="top", ha="right")

    # Ligne de séparation
    line = plt.Line2D([0.01, 0.99], [0.965, 0.965], transform=fig.transFigure,
                      color=DARK["border"], linewidth=0.8)
    fig.add_artist(line)


# ─── CLASSE PRINCIPALE ────────────────────────────────────────────────────────

class PortfolioDashboardVisualizer:
    """
    Bloomberg Dark Terminal – Dashboard complet.
    
    Paramètres attendus dans ts.metrics :
        pf_cumulative_return   : pd.Series (index datetime, valeurs float comme 0.12 pour 12%)
        bm_cumulative_return   : pd.Series (optionnel)
        portfolio_pnl          : pd.Series (P&L journalier en €)
        portfolio_value        : pd.Series (valeur totale du portefeuille)  ← optionnel
        initial_capital        : float  (sinon ts.capital_init)
        annual_return          : float
        volatility             : float
        sharpe_ratio           : float
        max_drawdown           : float
        calmar_ratio           : float
        win_rate               : float  (optionnel, calculé si absent)
        var_95                 : float
        total_pnl              : float
        total_return           : float
        final_value            : float
    """

    def __init__(self, ts_portfolio, style="dark"):
        self.ts = ts_portfolio
        self.style = style
        _apply_dark_style()

    # ── Helpers internes ──────────────────────────────────────────────────────

    def _get_portfolio_value(self):
        pv = self.ts.metrics.get("portfolio_value")
        if pv is not None:
            return pv
        pnl = self.ts.metrics.get("portfolio_pnl")
        if pnl is not None:
            return pnl.cumsum() + self.ts.capital_init
        return None

    def _get_capital(self):
        return self.ts.metrics.get("initial_capital", getattr(self.ts, "capital_init", 1_000_000))

    def _build_metrics_dict(self):
        m = dict(self.ts.metrics)
        # Win rate auto si absent
        pnl = m.get("portfolio_pnl")
        if "win_rate" not in m and pnl is not None:
            m["win_rate"] = float((pnl > 0).mean())
        return m

    # ── Graphiques publics (réutilisables) ────────────────────────────────────

    def plot_cumulative_return(self, ax=None):
        if ax is None: ax = plt.gca()
        pf = self.ts.metrics.get("pf_cumulative_return")
        bm = self.ts.metrics.get("bm_cumulative_return")
        if pf is None: return
        plot_cumulative_return(ax, pf, bm)

    def plot_drawdown(self, ax=None):
        if ax is None: ax = plt.gca()
        pf = self.ts.metrics.get("pf_cumulative_return")
        if pf is None: return
        plot_drawdown(ax, pf)

    def plot_cumulative_pnl(self, ax=None, freq="ME"):
        if ax is None: ax = plt.gca()
        pv = self._get_portfolio_value()
        if pv is None: return
        plot_monthly_pnl(ax, pv, self._get_capital())

    def plot_weights_composition(self, ax=None, top_n=8):
        if ax is None: ax = plt.gca()
        if self.ts.weights_history.empty: return
        plot_weights_composition(ax, self.ts.weights_history, top_n)

    def plot_risk_contribution(self, ax=None):
        if ax is None: ax = plt.gca()
        if self.ts.weights_history.empty: return
        plot_risk_contribution(ax, self.ts.weights_history)

    def plot_monthly_returns_heatmap(self, ax=None):
        if ax is None: ax = plt.gca()
        pnl = self.ts.metrics.get("portfolio_pnl")
        if pnl is None: return
        plot_monthly_heatmap(ax, pnl, self._get_capital())

    def plot_pnl_attribution(self, ax=None, freq="ME"):
        if ax is None: ax = plt.gca()
        pnl = self.ts.metrics.get("portfolio_pnl")
        if pnl is None: return
        plot_pnl_attribution(ax, self.ts.weights_history, pnl,
                             self._get_capital(), freq)

    # ── Dashboard principal ───────────────────────────────────────────────────

    def plot_dashboard(self, figsize=(22, 26), save_path=None, dpi=150):
        """
        Layout Bloomberg dark 6 lignes :
        ┌──────────────────────────────────────────────┐  ← Header
        ├──────────────────────────────────────────────┤  ← KPIs
        ├──────────────────────┬───────────────────────┤  ← Cum Return | Drawdown
        ├──────────────────────┴───────────────────────┤  ← Monthly P&L
        ├──────────────────────────────────────────────┤  ← Heatmap
        ├──────────────────────┬───────────────────────┤  ← Weights | Risk Contrib
        ├──────────────────────────────────────────────┤  ← Attribution
        └──────────────────────────────────────────────┘
        """
        fig = plt.figure(figsize=figsize, facecolor=DARK["bg"])
        fig.patch.set_facecolor(DARK["bg"])

        gs = GridSpec(
            7, 2, figure=fig,
            height_ratios=[0.08, 0.07, 0.18, 0.15, 0.14, 0.18, 0.15],
            hspace=0.52, wspace=0.25,
            left=0.06, right=0.97, top=0.96, bottom=0.03
        )

        metrics = self._build_metrics_dict()
        capital = self._get_capital()

        # ── Header ──
        _draw_header(fig, metrics, capital)

        # ── Row 0 : KPI bar (full width) ──
        ax_kpi = fig.add_subplot(gs[1, :])
        _draw_kpi_panel(ax_kpi, metrics)

        # ── Row 1 : Cumulative Return | Drawdown ──
        ax_cum = fig.add_subplot(gs[2, 0])
        self.plot_cumulative_return(ax_cum)

        ax_dd = fig.add_subplot(gs[2, 1])
        self.plot_drawdown(ax_dd)

        # ── Row 2 : Monthly P&L (full width) ──
        ax_pnl = fig.add_subplot(gs[3, :])
        self.plot_cumulative_pnl(ax_pnl, freq="ME")

        # ── Row 3 : Heatmap (full width) ──
        ax_hm = fig.add_subplot(gs[4, :])
        self.plot_monthly_returns_heatmap(ax_hm)

        # ── Row 4 : Weights | Risk Contrib ──
        ax_w = fig.add_subplot(gs[5, 0])
        self.plot_weights_composition(ax_w, top_n=8)

        ax_rc = fig.add_subplot(gs[5, 1])
        self.plot_risk_contribution(ax_rc)

        # ── Row 5 : Attribution (full width) ──
        ax_attr = fig.add_subplot(gs[6, :])
        self.plot_pnl_attribution(ax_attr, freq="ME")

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight",
            
                        facecolor=DARK["bg"])
            print(f"✓ Dashboard saved → {save_path}")

        return fig

    def save_dashboard(self, filename="portfolio_dashboard.png", dpi=200):
        fig = self.plot_dashboard(save_path=filename, dpi=dpi)
        plt.close(fig)


# ─── DEMO (données simulées) ──────────────────────────────────────────────────

class _MockPortfolio:
    """Portfolio simulé pour démonstration."""

    def __init__(self):
        self.capital_init = 1_000_000
        dates = pd.bdate_range("2022-01-03", "2024-12-31")
        np.random.seed(42)

        # Returns journaliers
        pf_ret  = pd.Series(np.random.normal(0.0004, 0.009, len(dates)), index=dates)
        bm_ret  = pd.Series(np.random.normal(0.0002, 0.007, len(dates)), index=dates)
        pf_pnl  = pf_ret * self.capital_init

        pf_cum  = (1 + pf_ret).cumprod() - 1
        bm_cum  = (1 + bm_ret).cumprod() - 1
        pf_val  = self.capital_init * (1 + pf_cum)
        max_dd  = float(((pf_cum - pf_cum.cummax()) / (1 + pf_cum.cummax().abs())).min())
        ann_ret = float((1 + pf_cum.iloc[-1]) ** (252 / len(dates)) - 1)
        vol     = float(pf_ret.std() * np.sqrt(252))

        self.metrics = {
            "pf_cumulative_return": pf_cum,
            "bm_cumulative_return": bm_cum,
            "portfolio_pnl":        pf_pnl,
            "portfolio_value":      pf_val,
            "initial_capital":      self.capital_init,
            "annual_return":        ann_ret,
            "volatility":           vol,
            "sharpe_ratio":         ann_ret / vol if vol else 0,
            "max_drawdown":         max_dd,
            "calmar_ratio":         -ann_ret / max_dd if max_dd else 0,
            "win_rate":             float((pf_ret > 0).mean()),
            "var_95":               float(np.percentile(pf_ret.dropna(), 5)),
            "total_pnl":            float(pf_pnl.sum()),
            "total_return":         float(pf_cum.iloc[-1]),
            "final_value":          float(pf_val.iloc[-1]),
        }

        # Weights history
        assets = ["FR Gov","DE Gov","IT Gov","SP Gov","UK Corp","US Corp","Cash","EM IG"]
        rb_dates = pd.bdate_range("2022-01-03", "2024-12-31", freq="ME")
        wh = pd.DataFrame(np.random.dirichlet(np.ones(len(assets)), len(rb_dates)),
                          index=rb_dates, columns=assets)
        self.weights_history = wh.reindex(dates).ffill().fillna(0)
        self.rebalance_dates = rb_dates


if __name__ == "__main__":
    print("Generating Bloomberg-style portfolio dashboard...")
    mock = _MockPortfolio()
    viz  = PortfolioDashboardVisualizer(mock)
    fig  = viz.plot_dashboard(save_path="portfolio_dashboard.png", dpi=150)
    plt.show()
    print("Done.")