import numpy as np


def calculate_cash_erosion_paths(cpi_paths, initial_capital, horizon, tolerance_percent=5):
    """Simulates what `initial_capital` would be worth, in today's purchasing
    power, if left as uninvested cash for `horizon` months using the exact
    same simulated CPI paths that drive the bond fanchart, so it's directly
    comparable.

    cpi_paths: (num_sim, total_sim_horizon) array of simulated *annual* CPI
        rates (decimal, e.g. 0.03 for 3%), as produced by simulate_paths.
    horizon: number of months to project the cash erosion over. Callers
        should pass at least the longest strategy horizon they need; the
        returned monthly paths can then be sliced/indexed per strategy,
        since each month's percentile is computed independently of horizon.

    Each month's annual CPI rate is converted to an equivalent monthly price
    growth factor via (1 + r)^(1/12), compounded forward to build a cumulative
    price index. Real cash value = initial_capital / cumulative_price_index.
    Month 0 has no erosion yet (cash starts at full nominal value).

    Returns a dict with "mean", "worst" (tolerance_percent, i.e. the highest-
    inflation / lowest real-value tail), and "best" (100-tolerance_percent,
    lowest-inflation tail) paths, each an array of length `horizon`.
    """
    cpi_slice = cpi_paths[:, :horizon]

    monthly_factor = (1.0 + cpi_slice) ** (1.0 / 12.0)
    cum_index = np.cumprod(monthly_factor, axis=1)

    # shift so index at month 0 is 1.0 (no erosion has happened yet)
    ones = np.ones((cpi_slice.shape[0], 1))
    cum_index_full = np.concatenate([ones, cum_index[:, :-1]], axis=1)

    real_value = initial_capital / cum_index_full

    return {
        "mean": np.mean(real_value, axis=0),
        "worst": np.percentile(real_value, tolerance_percent, axis=0),
        "best": np.percentile(real_value, 100 - tolerance_percent, axis=0),
    }


def cash_erosion_summary_at(cash_paths, month_idx, initial_capital):
    """Extracts the mean/worst/best real cash value (and the implied loss vs.
    the original nominal capital) at a specific month from paths returned by
    calculate_cash_erosion_paths. month_idx is 0-indexed (month 1 of a
    strategy's horizon is index 0)."""
    idx = max(0, min(month_idx, len(cash_paths["mean"]) - 1))

    mean_v = float(cash_paths["mean"][idx])
    worst_v = float(cash_paths["worst"][idx])
    best_v = float(cash_paths["best"][idx])

    return {
        "mean_real_value": mean_v,
        "worst_real_value": worst_v,
        "best_real_value": best_v,
        "mean_loss": initial_capital - mean_v,
        "worst_loss": initial_capital - worst_v,
        "best_loss": initial_capital - best_v,
    }
