import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec
import seaborn as sns



class PortfolioDashboardVisualizer:
    def __init__(self, ts_portfolio, dark = False):
        """
        ts_portfolio : instance de TimeSeriesPortfolio
        dark : active un thème sombre propre
        """
        self.ts = ts_portfolio
        
        if dark:
            plt.style.use("dark_background")
            sns.set(style="darkgrid")
        else:
            sns.set(style="whitegrid")

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------

    def _get_or_compute_portfolio_value(self):
        """
        Renvoie la valeur du portefeuille (si absente → reconstruct via PnL cumulés)
        """
        pf_value = self.ts.metrics.get("portfolio_value")
        pf_pnl = self.ts.metrics.get("portfolio_pnl")
        capital = self.ts.metrics.get("initial_capital", self.ts.capital_init)

        if pf_value is None or pf_value.empty:
            if pf_pnl is not None:
                return capital + pf_pnl.cumsum()
            return None
        return pf_value

    # -------------------------------------------------------------------------
    # CUMULATIVE RETURN
    # -------------------------------------------------------------------------
    def plot_cumulative_return(self, ax=None):
        pf = self.ts.metrics.get("pf_cumulative_return")
        bm = self.ts.metrics.get("bm_cumulative_return")

        if pf is None or bm is None:
            return

        if ax is None:
            ax = plt.gca()

        ax.plot(pf, lw=2, label="Portfolio")
        ax.plot(bm, lw=2, linestyle='--', label="Benchmark")
        ax.set_title("Cumulative Return")
        ax.set_ylabel("Cumulative Return")
        ax.grid(alpha=0.3)
        ax.legend()

    # -------------------------------------------------------------------------
    # VALUE (portfolio vs benchmark)
    # -------------------------------------------------------------------------
    def plot_value(self, ax=None):
        pf = self._get_or_compute_portfolio_value()
        bm = self.ts.metrics.get("benchmark_value")

        if pf is None or bm is None:
            return

        if ax is None:
            ax = plt.gca()

        ax.plot(pf, lw=2, label="Portfolio Value")
        ax.plot(bm, lw=2, linestyle='--', label="Benchmark Value")
        ax.set_ylabel("Value (€)")
        ax.set_title("Portfolio & Benchmark Value (constant exposure)")
        ax.grid(alpha=0.3)
        ax.legend()

    # -------------------------------------------------------------------------
    # HELPERS FOR PNL / RESAMPLING
    # -------------------------------------------------------------------------
    def _compute_resampled_pnl(self, freq='W'):
        """
        Retourne pf_monthly, bm_monthly, pf_monthly_diff
        """
        pf_value = self._get_or_compute_portfolio_value()
        bm_value = self.ts.metrics.get("benchmark_value")
        capital = self.ts.metrics.get("initial_capital", self.ts.capital_init)

        if pf_value is None or bm_value is None:
            return None, None, None

        pf_m = pf_value.resample(freq).last().ffill() - capital
        bm_m = bm_value.resample(freq).last().ffill() - capital

        pf_diff = pf_m.diff().fillna(0)

        return pf_m, bm_m, pf_diff

    # -------------------------------------------------------------------------
    # CUMULATIVE PNL with bars
    # -------------------------------------------------------------------------
    def plot_cumulative_pnl(self, ax=None, show_benchmark=False):
        pf_monthly, bm_monthly, pf_monthly_diff = self._compute_resampled_pnl(freq='W')

        if pf_monthly is None:
            return

        if ax is None:
            ax = plt.gca()

        # --- Bars (green/red) ---
        for i in range(1, len(pf_monthly)):
            start = pf_monthly.iloc[i - 1]
            end = pf_monthly.iloc[i]
            color = 'green' if end >= start else 'red'
            ax.bar(
                pf_monthly.index[i],
                end - start,
                bottom=start,
                width=20,
                color=color
                
            )

        # --- Optional benchmark line ---
        if show_benchmark and bm_monthly is not None:
            ax.plot(bm_monthly.index, bm_monthly, lw=2, linestyle='--', label='Benchmark cum PnL', color = 'orange')

        ax.set_ylabel("Cumulative PnL (€)")
        ax.set_title(f"Cumulative PnL – Exposure: {self.ts.metrics.get('initial_capital')}€")
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:,.0f}'))
        ax.legend()
    
    def plot_big_cumulative_pnl_pf_only(self, ax=None):
        """
        Version lisible et 'grosse' du cumulative PnL du portefeuille.
        - PF uniquement
        - Barres vert/rouge collées
        - Ligne cumulative
        """

        pf_value = self._get_or_compute_portfolio_value()
        capital = self.ts.metrics.get("initial_capital", self.ts.capital_init)

        if pf_value is None:
            return

        # Resample weekly for smoother display
        pf_monthly = pf_value.resample('ME').last().ffill()
        pf_pnl = pf_monthly - capital
        pf_diff = pf_pnl.diff().fillna(0)

        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 6))

        # --- Barres rouge/vert ---
        for i in range(1, len(pf_pnl)):
            start = pf_pnl.iloc[i - 1]
            end = pf_pnl.iloc[i]
            color = 'green' if end >= start else 'red'
            ax.bar(
                pf_pnl.index[i],
                end - start,
                bottom=start,
                width=20,
                color=color,
                alpha=0.8
            )

        # --- Ligne cumulative ---
        """
        ax.plot(
            pf_pnl.index,
            pf_pnl.values,
            linewidth=2.5,
            color='white' if plt.rcParams['axes.facecolor'] != 'white' else 'blue',
            label="Portfolio cumulative PnL"
        )
        """
        
        ax.set_title("Portfolio Cumulative Monthly PnL ", fontsize=15, fontweight='bold')
        ax.set_ylabel("PnL (€)")
        ax.grid(alpha=0.3)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:,.0f}'))
        ax.legend()
    
    def plot_pnl_attribution(self, ax=None, freq='ME'):
        """
        Attribution du PnL par actif/pays.
        freq : 'ME' monthly, 'W' weekly
        """
        pf_pnl = self.ts.metrics.get("portfolio_pnl")
        exposures = self.ts.exposures_history
        R = self.ts.R[exposures.columns]

        if pf_pnl is None or exposures.empty:
            print("Attribution impossible : pas de PnL ou d'exposures.")
            return

        # Align
        R = R.loc[pf_pnl.index].fillna(0)
        exposures = exposures.reindex(self.ts.rebalance_dates[:-1]).ffill().reindex(pf_pnl.index, method='ffill')

        # PnL par actif
        pnl_assets = R * exposures
        pnl_assets_resampled = pnl_assets.resample(freq).sum()

        if ax is None:
            ax = plt.gca()

        pnl_assets_resampled.plot(kind='bar', stacked=True, ax=ax, figsize=(10,5))
        ax.set_title(f"PnL Attribution ({freq})")
        ax.set_ylabel("PnL (€)")
        ax.grid(alpha=0.3)

    # -------------------------------------------------------------------------
    # MONTHLY VALUE CHANGE
    # -------------------------------------------------------------------------
    def plot_monthly_value(self, ax=None):
        pf_value = self._get_or_compute_portfolio_value()
        bm_value = self.ts.metrics.get("benchmark_value")

        if pf_value is None:
            return

        # Resample end-of-month
        pf_m = pf_value.resample('ME').last()
        bm_m = bm_value.resample('ME').last() if bm_value is not None else None

        pf_diff = pf_m.diff().fillna(0)

        if ax is None:
            ax = plt.gca()

        colors = ['green' if x >= 0 else 'red' for x in pf_diff]

        ax.bar(pf_m.index, pf_diff.values, color=colors, width=20)
        ax.set_ylabel("Monthly Change (€)")
        ax.set_title("Monthly Portfolio Value Change")
        ax.grid(alpha=0.3)
        ax.legend(["Monthly PnL"])

        # Cumulative line
        ax2 = ax.twinx()
        ax2.plot(pf_m.index, pf_m.values, lw=2, label='Portfolio Value')
        if bm_m is not None:
            ax2.plot(bm_m.index, bm_m.values, lw=2, linestyle='--', label='Benchmark Value')

        ax2.set_ylabel("Value (€)")
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:,.0f}'))
        ax2.legend(loc='upper right')

    # -------------------------------------------------------------------------
    # EXPOSURES
    # -------------------------------------------------------------------------
    def plot_exposures(self, ax=None):
        if self.ts.exposures_history.empty:
            return

        gross = np.abs(self.ts.exposures_history).sum(axis=1)
        net = self.ts.exposures_history.sum(axis=1)

        if ax is None:
            ax = plt.gca()

        ax.plot(gross, lw=2, label="Gross Exposure")
        ax.plot(net, lw=2, linestyle='--', label="Net Exposure")
        ax.set_ylabel("Exposure (€ DV01)")
        ax.set_title("Gross vs Net Exposure")
        ax.grid(alpha=0.3)
        ax.legend()

    
    def plot_exposures_by_country(self, ax=None):
        country_expo = np.abs(self.ts.exposition_by_country())
        country_expo.plot(kind='area', stacked=True, ax=ax, alpha=0.7, legend=True)


        ax.set_ylabel("Exposition in Euros")
        ax.set_title("Exposition by Country")
        ax.legend(title="Country", loc="upper left")
        ax.grid(alpha=0.3)

    def plot_exposures_by_mat(self, ax=None):
        mat_expo = np.abs(self.ts.exposition_by_mat())
        
        mat_expo.plot(kind='area', stacked=True, ax=ax, alpha=0.7, legend=True)

        ax.set_ylabel("Exposition in Euros")
        ax.set_title("Exposition by mat")
        ax.legend(title="Maturity", loc="upper left")
        ax.grid(alpha=0.3)

    # -------------------------------------------------------------------------
    # WEIGHTS
    # -------------------------------------------------------------------------
    def plot_weights(self, ax=None, abs_weights=True):
        if self.ts.weights_history.empty:
            return

        df = np.abs(self.ts.weights_history) if abs_weights else self.ts.weights_history
        
        if ax is None:
            ax = plt.gca()

        # Trace toutes les séries (toutes les couleurs seront visibles)
        df.plot(kind='area', stacked=True, ax=ax, alpha=0.7, legend=False)

        # --- Calcul du top 5 ---
        top5 = df.mean().nlargest(5).index.tolist()

        # --- Récupération handles + labels générés automatiquement ---
        handles, labels = ax.get_legend_handles_labels()

        # Le plot n'a pas encore de légende → on génère les handles de façon cohérente
        full_handles = ax.get_children()[::-1]  # matplotlib stocke dans l'ordre inverse
        patch_handles = [h for h in full_handles if hasattr(h, "get_facecolor")][:len(df.columns)]

        # Mapping produit -> handle
        handle_map = dict(zip(df.columns, patch_handles))

        # --- Construire une légende ne contenant que le top-5 ---
        top5_handles = [handle_map[p] for p in top5]

        ax.legend(top5_handles, top5, title="Top 5 produits", loc="upper left")

        ax.set_ylabel("Weights" + (" (abs)" if abs_weights else ""))
        ax.set_title("Portfolio Weights")
        ax.grid(alpha=0.3)

    # -------------------------------------------------------------------------
    # METRIC TABLE
    # -------------------------------------------------------------------------
    def print_metrics_table(self, ax=None):
        metrics_table = {
            k: v for k, v in self.ts.metrics.items()
            if np.isscalar(v) or isinstance(v, (int, float))
        }

        df = pd.DataFrame.from_dict(metrics_table, orient='index', columns=["Value"])
        df.index.name = "Metric"

        if ax is None:
            ax = plt.gca()

        ax.axis('off')
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            rowLabels=df.index,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.4)
    
    # -----------------------------------------------------
    # Max draw down 
    # -----------------------------------------------------
    def plot_drawdown(self, ax=None):
        dd = self.ts.metrics.get("drawdown")
        if dd is None or dd.empty:
            print("Drawdown non disponible.")
            return

        if ax is None:
            ax = plt.gca()

        ax.plot(dd, color='red', lw=2)
        ax.fill_between(dd.index, dd, 0, color='red', alpha=0.3)
        ax.set_title("Portfolio Drawdown")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(alpha=0.3)
    

    # -------------------------------------------------------------------------
    #Rsik contribution 
    # ------------------------------------------------------------------------
    def plot_risk_contribution(self, ax=None):
        """
        Contribution au risque ex-post par actif : RC_i = w_i * (Σ_cov_row_i) / vol_pf
        """
        w = self.ts.weights_history
        R = self.ts.R[self.ts.weights_history.columns]

        if w.empty or R.empty:
            print("Risk contribution impossible : poids ou retours manquants.")
            return

        # Covariance matrix ex-post
        cov = R.cov()
        avg_w = w.mean()  # moyenne des poids dans le temps
        
        pf_vol = np.sqrt(avg_w @ cov @ avg_w)

        if pf_vol == 0:
            print("Volatilité PF nulle → RC impossible.")
            return

        rc = avg_w * (cov @ avg_w) / pf_vol
        rc = rc.sort_values(ascending=False)

        if ax is None:
            ax = plt.gca()

        rc.plot(kind='bar', ax=ax)
        ax.set_title("Risk Contribution (Ex-post)")
        ax.set_ylabel("Contribution au risque (%)")
        ax.grid(alpha=0.3)

    # -------------------------------------------------------------------------
    # FULL DASHBOARD
    # -------------------------------------------------------------------------
    def plot_dashboard(self):
        fig = plt.figure(figsize=(15, 28))  # un peu plus haut pour tout tenir
        gs = GridSpec(8, 2, figure=fig, width_ratios=[2, 2])

        # Row 1
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_cumulative_pnl(ax1, show_benchmark=True)
        self.plot_value(ax2)

        # Row 2
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        self.plot_weights(ax3)
        self.plot_exposures(ax4)

        #Row 3 — 
        ax_1 = fig.add_subplot(gs[2, 0])
        ax_2 = fig.add_subplot(gs[2, 1])
        self.plot_exposures_by_country(ax_1)
        self.plot_exposures_by_mat(ax_2)

        # Row 4
        ax5 = fig.add_subplot(gs[3, 1])

        self.print_metrics_table(ax5)

        # Row 5
        ax6 = fig.add_subplot(gs[4, :])
        self.plot_monthly_value(ax6)

        # Row 6 — Drawdown
        ax_dd = fig.add_subplot(gs[5, :])
        self.plot_drawdown(ax_dd)



        # Row 7-8 — Big cumulative PnL PF only
        ax7 = fig.add_subplot(gs[6:, :])
        self.plot_big_cumulative_pnl_pf_only(ax7)

        plt.tight_layout()
        plt.show()