import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ensure_results_dir() -> str:
    """Ensure the results directory exists."""
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

# Config
ANALYSIS_LOG = os.path.join(ensure_results_dir(), "analysis.txt")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def log_output(text: str, sep: bool = True) -> None:
    """Log to file and stdout."""
    with open(ANALYSIS_LOG, "a", encoding="utf-8") as f:
        f.write(text + "\n")
        if sep:
            f.write("\n")
    logger.info(text)
    if sep:
        logger.info("")

def clear_results_file() -> None:
    """Clear the analysis log at the start of the run."""
    with open(ANALYSIS_LOG, "w", encoding="utf-8") as f:
        f.write("========== Property Analysis Results ==========" + "\n\n")

def read_data(filepath: str, sheet_name: str = "Sheet1") -> pd.DataFrame:
    """Read property data from Excel."""
    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
    except Exception as e:
        logger.error(f"Failed to read Excel: {e}")
        raise
    logger.debug(f"Columns in loaded file: {df.columns.tolist()}")
    df.columns = [c.strip() for c in df.columns]
    # If 'PSF' column exists, clean it up
    if 'PSF' in df.columns:
        # Remove commas, spaces, and convert to numeric
        df['PSF'] = df['PSF'].astype(str).str.replace(',', '').str.strip()
        df['PSF'] = pd.to_numeric(df['PSF'], errors='coerce')
    # If 'Date' column exists, parse to datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    logger.debug(f"First 5 rows after cleaning:\n{df.head()}")
    return df

def filter_resale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the DataFrame for resale transactions and ensures PSF is numeric.
    """
    df = df[df["Type of Sale"] == "Resale"].copy()
    df["PSF"] = pd.to_numeric(df["PSF"], errors="coerce")
    return df

def preprocess_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the 'Date' column to datetime.
    """
    df["Date"] = pd.to_datetime(df["Date"])
    return df

def get_project_start_dates(df: pd.DataFrame) -> pd.Series:
    """
    Gets the earliest transaction date for each project.
    """
    return df.groupby("Project")["Date"].min()

def get_3m_avg_psf(df: pd.DataFrame, project: str, start_date: pd.Timestamp) -> float:
    """
    Computes the 3-month average PSF for a project from a given start date.
    Adds debug logs to show how many records are found and the computed base PSF.
    Also prints sample data for Caspian for root cause analysis.
    """
    mask = (
        (df["Project"] == project) &
        (df["Date"] >= start_date) &
        (df["Date"] < start_date + pd.DateOffset(months=3))
    )
    psf_values = df.loc[mask, "PSF"]
    # Extra debug for Caspian
    if project.lower() == 'caspian':
        logger.debug(f"[DEBUG] Caspian 3m window data (start={start_date.date()}):")
        logger.debug("\n" + str(df.loc[mask, [col for col in df.columns if col in ['Date','PSF','Project']]].head(10)))
        logger.debug(f"[DEBUG] Caspian PSF dtype: {psf_values.dtype}, NaN count: {psf_values.isna().sum()}, Values: {psf_values.values}")
    avg_psf = psf_values.mean() # type: ignore
    log_output(f"[DEBUG] get_3m_avg_psf: project='{project}', start_date={start_date.date()}, count={psf_values.shape[0]}, avg_psf={avg_psf}")
    return avg_psf

def compute_project_bases(df: pd.DataFrame, projects: np.ndarray, start_date: pd.Timestamp) -> dict[str, float]:
    """
    Computes the 3-month average PSF base for each project from a given start date.
    """
    return {
        project: get_3m_avg_psf(df, project, start_date)
        for project in projects
    }

def compute_project_indices(
    df: pd.DataFrame,
    projects: np.ndarray,
    project_bases: dict[str, float],
    start_date: pd.Timestamp
) -> dict[str, pd.Series]:
    """
    Computes normalized and smoothed monthly PSF indices for each project.
    """
    df["MonthPeriod"] = df["Date"].dt.to_period("M")
    project_indices: dict[str, pd.Series] = {}
    for project in projects:
        proj_df = df[df["Project"] == project].copy()
        proj_df = proj_df[proj_df["Date"] >= start_date]
        if proj_df.empty:
            log_output(f"[DEBUG] Skipping project '{project}': no data after {start_date.date()}")
            continue
        monthly = proj_df.groupby("MonthPeriod")["PSF"].mean().sort_index()
        base = project_bases[project]
        if pd.isna(base) or base == 0:
            log_output(f"[DEBUG] Skipping project '{project}': 3-month base PSF is NaN or 0 after {start_date.date()}")
            continue
        index = (monthly / base) * 100
        index_smooth = index.rolling(window=3, min_periods=1).mean()
        project_indices[project] = index_smooth
    return project_indices

def plot_indices(
    project_indices: dict[str, pd.Series],
    title: str,
    ylabel: str,
    base_year: int | None = None,
    filename: str | None = None
) -> None:
    """
    Plots the normalized PSF indices for all projects and saves to results folder.
    """
    plt.figure(figsize=(12, 7))
    for project, index in project_indices.items():
        plt.plot(index.index.to_timestamp(), index.values, label=project, linewidth=2, alpha=0.85) # type: ignore
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    results_dir = ensure_results_dir()
    if filename is None:
        filename = title.replace(" ", "_").replace("(", "").replace(")", "").replace("'", "") + ".png"
    filepath = os.path.join(results_dir, filename)
    plt.savefig(filepath)
    plt.close()

def plot_indices_for_years(
    df: pd.DataFrame,
    projects: np.ndarray,
    years: list[int]
) -> None:
    """
    Plots normalized indices for each project, using different base years.
    """
    for year in years:
        start_date = pd.Timestamp(f"{year}-01-01")
        project_bases_year = compute_project_bases(df, projects, start_date)
        project_indices_year = compute_project_indices(df, projects, project_bases_year, start_date)
        plot_indices(
            project_indices_year,
            f"Condo PSF Price Index (Resale) - Normalized to Jan {year} (Smoothed)",
            f"Price Index (Base=100 at Jan {year})",
            filename=f"psf_index_jan_{year}.png"
        )

def compute_annual_indices(project_indices: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """
    Computes annual average index for each project.
    """
    annual_indices: dict[str, pd.Series] = {}
    for project, index in project_indices.items():
        idx = index.copy()
        # Only convert PeriodIndex to DatetimeIndex
        if isinstance(idx.index, pd.PeriodIndex):
            idx.index = idx.index.to_timestamp()
        annual = idx.resample('YE').mean()  # Use 'YE' instead of 'A'
        annual.index = annual.index.map(lambda x: x.year)
        annual_indices[project] = annual
    return annual_indices

def compute_annual_returns(
    annual_indices: dict[str, pd.Series]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes year-on-year percent returns for each project.
    """
    annual_df = pd.DataFrame(annual_indices)
    annual_returns = annual_df.pct_change(fill_method=None) * 100  # Avoid deprecated fill_method
    return annual_returns, annual_df

def compute_median_returns(returns_df: pd.DataFrame) -> pd.Series:
    """
    Computes the median return across all projects for each period.
    """
    return returns_df.median(axis=1)

def analyze_performance_transitions(
    returns_df: pd.DataFrame,
    median_returns: pd.Series
) -> tuple[int, int, int, int, dict[str, dict[str, int]]]:
    """
    Analyzes transitions between underperformance and overperformance year-on-year.
    """
    under_to_over = 0
    over_to_under = 0
    total_under = 0
    total_over = 0
    property_stats: dict[str, dict[str, int]] = {}
    for project in returns_df.columns:
        returns = returns_df[project]
        u2o = 0
        o2u = 0
        t_under = 0
        t_over = 0
        for year in range(returns.index.min(), returns.index.max()):
            this_year = returns.loc[year]
            next_year = returns.loc[year + 1] if (year + 1) in returns.index else None
            if pd.isna(this_year) or pd.isna(next_year):
                continue
            median_this = median_returns.loc[year]
            median_next = median_returns.loc[year + 1]
            if this_year < median_this:
                total_under += 1
                t_under += 1
                if next_year > median_next:
                    under_to_over += 1
                    u2o += 1
            elif this_year > median_this:
                total_over += 1
                t_over += 1
                if next_year < median_next:
                    over_to_under += 1
                    o2u += 1
        property_stats[project] = {
            "under_to_over": u2o,
            "over_to_under": o2u,
            "total_under": t_under,
            "total_over": t_over
        }
    return under_to_over, over_to_under, total_under, total_over, property_stats

def plot_annual_returns(annual_returns: pd.DataFrame) -> None:
    """
    Plots a grouped bar chart of annualized returns for each project and saves to results folder.
    """
    plt.figure(figsize=(16, 8))
    bar_width = 0.8 / len(annual_returns.columns)
    years = annual_returns.index.astype(str)
    x = np.arange(len(years))
    for i, project in enumerate(annual_returns.columns):
        plt.bar(
            x + i * bar_width,
            annual_returns[project],
            width=bar_width,
            label=project,
            align='center'
        )
    plt.xlabel("Year")
    plt.ylabel("Annualized Return (%)")
    plt.title("Year-on-Year Annualized Return by Property")
    plt.xticks(x + bar_width * (len(annual_returns.columns) - 1) / 2, list(years), rotation=45)
    plt.legend()
    plt.tight_layout()
    results_dir = ensure_results_dir()
    plt.savefig(os.path.join(results_dir, "annualized_returns.png"))
    plt.close()

def compute_quarterly_indices(project_indices: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """
    Computes quarterly average index for each project.
    """
    quarterly_indices: dict[str, pd.Series] = {}
    for project, index in project_indices.items():
        idx = index.copy()
        if not isinstance(idx.index, pd.DatetimeIndex):
            idx.index = idx.index.to_timestamp() # type: ignore
        quarterly = idx.resample('QE').mean()
        quarterly_indices[project] = quarterly
    return quarterly_indices

def compute_quarterly_returns(
    quarterly_indices: dict[str, pd.Series]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes quarter-on-quarter percent returns for each project.
    """
    quarterly_df = pd.DataFrame(quarterly_indices)
    quarterly_returns = quarterly_df.pct_change() * 100
    return quarterly_returns, quarterly_df

def analyze_quarterly_performance_transitions(
    quarterly_returns: pd.DataFrame,
    median_q_returns: pd.Series
) -> tuple[int, int, int, int, dict[str, dict[str, int]]]:
    """
    Analyzes transitions between underperformance and overperformance quarter-on-quarter.
    """
    q_under_to_over = 0
    q_over_to_under = 0
    q_total_under = 0
    q_total_over = 0
    q_property_stats: dict[str, dict[str, int]] = {}
    for project in quarterly_returns.columns:
        returns = quarterly_returns[project]
        u2o = 0
        o2u = 0
        t_under = 0
        t_over = 0
        for i in range(len(returns.index) - 1):
            this_q = returns.iloc[i]
            next_q = returns.iloc[i + 1]
            if pd.isna(this_q) or pd.isna(next_q):
                continue
            median_this = median_q_returns.iloc[i]
            median_next = median_q_returns.iloc[i + 1]
            if this_q < median_this:
                q_total_under += 1
                t_under += 1
                if next_q > median_next:
                    q_under_to_over += 1
                    u2o += 1
            elif this_q > median_this:
                q_total_over += 1
                t_over += 1
                if next_q < median_next:
                    q_over_to_under += 1
                    o2u += 1
        q_property_stats[project] = {
            "under_to_over": u2o,
            "over_to_under": o2u,
            "total_under": t_under,
            "total_over": t_over
        }
    return q_under_to_over, q_over_to_under, q_total_under, q_total_over, q_property_stats

def compute_full_history_quarterly_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes quarterly returns for each condo using all available data (not just since newest launch).
    """
    df = df.copy()
    df["Quarter"] = df["Date"].dt.to_period("Q")
    quarterly_psf = df.groupby(["Project", "Quarter"])["PSF"].mean().unstack("Project")
    quarterly_psf.index = quarterly_psf.index.to_timestamp() # type: ignore
    quarterly_returns = quarterly_psf.pct_change() * 100
    return quarterly_returns

def compute_quarterly_winners_from_returns(quarterly_returns: pd.DataFrame) -> pd.DataFrame:
    """
    For each quarter, determine which condo(s) had the highest return.
    Returns a DataFrame of shape (quarters, condos) with 1 if won, 0 otherwise.
    """
    winners = pd.DataFrame(0, index=quarterly_returns.index, columns=quarterly_returns.columns)
    for idx, row in quarterly_returns.iterrows():
        max_val = row.max()
        if pd.isna(max_val):
            continue
        for col in row.index[row == max_val].tolist():
            winners.at[idx, col] = 1
    return winners

def plot_quarterly_winners_matrix(
    winners: pd.DataFrame,
    quarterly_returns: pd.DataFrame | None = None
) -> None:
    """
    Plots a vertical heatmap of quarterly performance for each condo.
    Color is proportional to normalized quarter-on-quarter return (green=best, red=worst, white=median).
    Actual performance values for the quarters are stored inside the cells.
    Quarters where a condo has no data are greyed out.
    """
    import seaborn as sns
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    if quarterly_returns is None:
        raise ValueError("quarterly_returns DataFrame must be provided.")

    concise_idx = [f"{i.year % 100}Q{i.quarter}" for i in winners.index]
    total_wins = winners.sum().astype(int)
    winners_with_total = winners.copy()
    winners_with_total.loc['Total Wins'] = total_wins

    # Normalize returns per quarter (row-wise min-max scaling)
    heatmap_data = quarterly_returns.copy()
    for idx in heatmap_data.index:
        row = heatmap_data.loc[idx]
        min_val = row.min()
        max_val = row.max()
        if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
            heatmap_data.loc[idx] = np.nan  # All missing or constant, set to nan
        else:
            heatmap_data.loc[idx] = (row - min_val) / (max_val - min_val)

    heatmap_data.loc['Total Wins'] = np.nan

    # Prepare annotation with actual return values (rounded to 2 decimals)
    annot_data = quarterly_returns.copy().round(2).astype(str)
    annot_data = annot_data.replace("nan", "")
    annot_data.loc['Total Wins'] = ""
    # Mask for missing data: True where data is missing for a condo in a quarter
    mask = heatmap_data.isna()

    vmin = 0.0
    vmax = 1.0
    vcenter = 0.5
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    plt.figure(figsize=(max(8, len(winners.columns) * 0.7), max(8, len(winners) * 0.35)))
    ax = sns.heatmap(
        heatmap_data,
        cmap="RdYlGn",
        norm=norm,
        annot=annot_data.where(~mask, ""),
        fmt='',
        linewidths=0.2,
        linecolor='gray',
        mask=mask,
        xticklabels=True,
        yticklabels=concise_idx + ['Total Wins'],
        cbar_kws={'label': 'Normalized Quarterly Return (relative performance)'},
        square=False
    )
    # Set grey color for missing data
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color='lightgrey')
    # Re-apply mask to set grey color for missing data
    for (i, j), val in np.ndenumerate(mask.values):
        if val and i < len(heatmap_data) - 1:  # skip 'Total Wins' row
            ax.add_patch(Rectangle((j, i), 1, 1, color='lightgrey', lw=0))

    for j, condo in enumerate(winners.columns):
        ax.text(j + 0.5, len(heatmap_data) - 0.5, f"{total_wins[condo]}", 
                ha='center', va='center', color='black', fontsize=10, fontweight='bold')

    ax.set_xlabel("Condo Project")
    ax.set_ylabel("Quarter")
    # Shorten the title for better fit
    ax.set_title("Quarterly Performance Heatmap (Color=Relative Return, Annot=Actual Return)", fontsize=12)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    plt.tight_layout()
    results_dir = ensure_results_dir()
    plt.savefig(os.path.join(results_dir, "quarterly_winners_matrix.png"))
    plt.close()

def analyze_best_overall_purchase(
    df: pd.DataFrame,
    quarterly_returns: pd.DataFrame,
    winners: pd.DataFrame,
    analysis_type: str = "(Since Newest Launch)"
) -> None:
    """
    Analyzes which condo is probably the best purchase regardless of the quarter.
    Considers total quarters won, average quarterly return, and consistency.
    """
    log_output(f"===== Best Overall Purchase Analysis {analysis_type} =====")
    total_wins = winners.sum()
    avg_returns = quarterly_returns.mean()
    std_returns = quarterly_returns.std()
    summary = pd.DataFrame({
        "Total Wins": total_wins,
        "Avg Quarterly Return (%)": avg_returns,
        "Std Dev (%)": std_returns
    })
    # Rank by total wins, then by avg return, then by lowest std dev
    summary["Rank"] = (
        summary["Total Wins"].rank(ascending=False, method="min") +
        summary["Avg Quarterly Return (%)"].rank(ascending=False, method="min") +
        summary["Std Dev (%)"].rank(ascending=True, method="min")
    )
    # Explanation of rank score:
    # The Rank Score is the sum of each condo's rank in three categories:
    # (1) total quarters won (higher is better), 
    # (2) average quarterly return (higher is better),
    # (3) standard deviation of returns (lower is better, i.e., more consistent).
    # Lower total Rank Score means better overall performance across these metrics.
    best = summary.sort_values("Rank").iloc[0]
    best_project = summary.sort_values("Rank").index[0]
    log_output("Summary of condos (higher wins/returns, lower std dev is better):")
    for project, row in summary.sort_values("Rank").iterrows():
        log_output(
            f"  {project}: Wins={row['Total Wins']}, "
            f"Avg Return={row['Avg Quarterly Return (%)']:.2f}%, "
            f"Std Dev={row['Std Dev (%)']:.2f}%, "
            f"Rank Score={row['Rank']:.1f}"
        )
    log_output(f"\n🏆 Based on quarterly performance{analysis_type}, '{best_project}' is probably the best overall purchase: "
               f"{int(best['Total Wins'])} quarters won, "
               f"average quarterly return {best['Avg Quarterly Return (%)']:.2f}%, "
               f"std dev {best['Std Dev (%)']:.2f}% (lower is more consistent).")
    log_output("-" * 48)

def summarize_current_performance(
    annual_returns: pd.DataFrame,
    median_returns: pd.Series,
    property_stats: dict[str, dict[str, int]]
) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]]]:
    """
    Summarizes current underperforming and overperforming projects and their historical transition probabilities.
    """
    latest_year = annual_returns.index.max()
    median_latest = median_returns.loc[latest_year]
    underperforming_now = []
    overperforming_now = []
    for project in annual_returns.columns:
        recent_return = annual_returns.loc[latest_year, project]
        stats = property_stats[project]
        if pd.isna(recent_return):
            continue
        if recent_return < median_latest:
            underperforming_now.append((project, stats["under_to_over"], stats["total_under"]))
        elif recent_return > median_latest:
            overperforming_now.append((project, stats["over_to_under"], stats["total_over"]))
    return underperforming_now, overperforming_now

def print_performance_transitions(
    under_to_over: int,
    over_to_under: int,
    total_under: int,
    total_over: int,
    property_stats: dict[str, dict[str, int]]
) -> None:
    """
    Prints summary statistics for annual under/over performance transitions.
    """
    log_output("===== Annual Under/Over Performance Transitions =====")
    log_output(f"Transitions from underperformance to overperformance (next year): {under_to_over} / {total_under}")
    log_output(f"Transitions from overperformance to underperformance (next year): {over_to_under} / {total_over}")
    if total_under > 0:
        log_output("Probability of underperformance followed by overperformance: {:.1f}%".format(under_to_over / total_under * 100))
    if total_over > 0:
        log_output("Probability of overperformance followed by underperformance: {:.1f}%".format(over_to_under / total_over * 100))
    log_output("\n--- Per-property under/over performance transitions ---", sep=False)
    for project, stats in property_stats.items():
        log_output(f"{project}:\n  Under→Over: {stats['under_to_over']} / {stats['total_under']}\n  Over→Under: {stats['over_to_under']} / {stats['total_over']}", sep=False)
    log_output("-" * 48)

def print_quarterly_performance_transitions(
    q_under_to_over: int,
    q_over_to_under: int,
    q_total_under: int,
    q_total_over: int,
    q_property_stats: dict[str, dict[str, int]]
) -> None:
    """
    Prints summary statistics for quarterly under/over performance transitions.
    """
    log_output("===== Quarterly Under/Over Performance Transitions =====")
    log_output(f"Transitions from underperformance to overperformance (next quarter): {q_under_to_over} / {q_total_under}")
    log_output(f"Transitions from overperformance to underperformance (next quarter): {q_over_to_under} / {q_total_over}")
    if q_total_under > 0:
        log_output("Probability of underperformance followed by overperformance (quarter): {:.1f}%".format(q_under_to_over / q_total_under * 100))
    if q_total_over > 0:
        log_output("Probability of overperformance followed by underperformance (quarter): {:.1f}%".format(q_over_to_under / q_total_over * 100))
    log_output("\n--- Per-property quarterly under/over performance transitions ---", sep=False)
    for project, stats in q_property_stats.items():
        log_output(f"{project}:\n  Under→Over: {stats['under_to_over']} / {stats['total_under']}\n  Over→Under: {stats['over_to_under']} / {stats['total_over']}", sep=False)
    log_output("-" * 48)

def print_current_performance(
    underperforming_now: list[tuple[str, int, int]],
    overperforming_now: list[tuple[str, int, int]]
) -> None:
    """
    Prints the current underperforming and overperforming projects and their historical transition probabilities.
    """
    log_output("===== Current Year Forward-Looking Analysis =====")
    log_output("Properties currently UNDERPERFORMING (below median):", sep=False)
    if underperforming_now:
        for project, u2o, t_under in underperforming_now:
            prob = (u2o / t_under * 100) if t_under > 0 else 0
            log_output(f"  {project}: Under→Over {u2o}/{t_under} ({prob:.1f}%)", sep=False)
    else:
        log_output("  (None)", sep=False)
    log_output("")
    log_output("Properties currently OVERPERFORMING (above median):", sep=False)
    if overperforming_now:
        for project, o2u, t_over in overperforming_now:
            prob = (o2u / t_over * 100) if t_over > 0 else 0
            log_output(f"  {project}: Over→Under {o2u}/{t_over} ({prob:.1f}%)", sep=False)
    else:
        log_output("  (None)", sep=False)
    log_output("=" * 48)

def analyze(filepath: str) -> None:
    """Run the full property analysis pipeline."""
    clear_results_file()
    try:
        df = read_data(filepath)
    except Exception as e:
        logger.error(f"Could not load data for analysis: {e}")
        return
    if df.empty:
        logger.warning("Input data is empty. Skipping analysis.")
        return
    df = filter_resale(df)
    df = preprocess_dates(df)
    project_start_dates = get_project_start_dates(df)
    newest_start_date = project_start_dates.max()
    projects = df["Project"].unique()
    project_bases = compute_project_bases(df, projects, newest_start_date)
    project_indices = compute_project_indices(df, projects, project_bases, newest_start_date)
    log_output(f"[DEBUG] After compute_project_indices: {list(project_indices.keys())}")
    plot_indices(
        project_indices,
        "Condo PSF Price Index (Resale) - Normalized to Newest Project (Smoothed)",
        "Price Index (Base=100 at newest condo's launch)",
        filename="psf_index_newest_project.png"
    )
    years = [2018, 2019, 2020, 2021]
    plot_indices_for_years(df, projects, years)
    annual_indices = compute_annual_indices(project_indices)
    annual_returns, annual_df = compute_annual_returns(annual_indices)
    median_returns = compute_median_returns(annual_returns)
    under_to_over, over_to_under, total_under, total_over, property_stats = analyze_performance_transitions(annual_returns, median_returns)
    print_performance_transitions(under_to_over, over_to_under, total_under, total_over, property_stats)
    plot_annual_returns(annual_returns)
    quarterly_indices = compute_quarterly_indices(project_indices)
    quarterly_returns, quarterly_df = compute_quarterly_returns(quarterly_indices)
    median_q_returns = compute_median_returns(quarterly_returns)
    q_under_to_over, q_over_to_under, q_total_under, q_total_over, q_property_stats = analyze_quarterly_performance_transitions(quarterly_returns, median_q_returns)
    print_quarterly_performance_transitions(q_under_to_over, q_over_to_under, q_total_under, q_total_over, q_property_stats)
    quarterly_returns_full_history = compute_full_history_quarterly_returns(df)
    quaterly_winners_full_history = compute_quarterly_winners_from_returns(quarterly_returns_full_history)
    analyze_best_overall_purchase(df, quarterly_returns_full_history, quaterly_winners_full_history, analysis_type="(Full History)")
    plot_quarterly_winners_matrix(quaterly_winners_full_history, quarterly_returns_full_history)
    underperforming_now, overperforming_now = summarize_current_performance(annual_returns, median_returns, property_stats)
    print_current_performance(underperforming_now, overperforming_now)

if __name__ == "__main__":
    analyze("./scraped_data/all_properties.xlsx")
