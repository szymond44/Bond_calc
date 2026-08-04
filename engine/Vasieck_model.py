import pandas as pd
import numpy as np
import math
from scipy.stats import linregress


def calculate_shock(matrix, horizon, sim_number, mu=0, sigma=1, dt=1/12, dof=5):
    combined_shocks = []

    for col in matrix.columns:
        Z_parameter = np.random.normal(mu, sigma, (sim_number, horizon))
        V_parameter = np.random.chisquare(dof, (sim_number, horizon))

        current_shock = Z_parameter / np.sqrt(V_parameter / dof)

        combined_shocks.append(current_shock)

    return combined_shocks


def calculate_correlations(matrix, shocks):

    correlation_matrix = matrix.corr().to_numpy()

    correlation_values = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]

    L = np.linalg.cholesky(correlation_matrix)

    X_stacked = np.stack(shocks)
    W_stacked = np.tensordot(L, X_stacked, axes=1)

    return W_stacked, correlation_values


def _is_policy_rate(col_name):
    """Columns representing a central-bank policy rate (discrete, decision-driven)
    are simulated as a jump process instead of a continuous diffusion."""
    return 'nbp' in col_name.lower()


# NBP's official continuous inflation target is 2.5% (tolerance band 1.5%-3.5%).
# We anchor CPI's long-run OU mean here instead of the OLS-implied value, since
# the OLS intercept is pulled around by whichever regime dominates the sample
# (see calibration-window testing: it swung from ~3.0% to ~6.1% just by
# changing the start date). The target is what the simulation should converge
# to over long horizons under a credible inflation-targeting regime, which is
# the thing that actually matters for multi-year bond projections.
CPI_TARGET = 0.025

# Floor on the OU mean-reversion speed, expressed as a half-life. The raw OLS
# fit on this sample implies a ~5 year half-life, but that estimate is fragile
# (see calibration notes): the sample's *only* strong evidence of reversion is
# concentrated in the 2022-2024 shock/correction episode, so trimming or
# windowing it out makes the reversion estimate worse, not better. Rather than
# trust that fragile number, we floor the reversion speed using the standard
# ~1-2 year monetary policy transmission lag assumed in inflation-targeting
# literature. This keeps near-term dynamics (which the data does inform well)
# essentially untouched while preventing unrealistic multi-decade drift away
# from target at long horizons. This is a modeling assumption, not an
# empirical estimate -- adjust CPI_MIN_HALF_LIFE_YEARS if you have a better
# prior.
CPI_MIN_HALF_LIFE_YEARS = 2.0


def _calibrate_ou_params(col_name, series, dt, target_override=None, min_half_life_years=None):
    """Discretized Vasicek/OU calibration for continuously-diffusing series
    (e.g. inflation).

    target_override: if given, replaces the OLS-implied long-run mean `b`
        with this value (e.g. the official CPI target) -- see CPI_TARGET.
    min_half_life_years: if given, floors the mean-reversion speed `a` so it
        can't be slower than implied by this half-life, guarding against a
        fragile/understated OLS reversion estimate -- see
        CPI_MIN_HALF_LIFE_YEARS. `sigma` (short-term volatility) is left as
        the honest empirical estimate either way.
    """
    start_value = float(series.iloc[-1])

    X = series[:-1].values
    Y = np.diff(series.values)

    slope, intercept, _, _, _ = linregress(X, Y)
    a_ols = float(-slope / dt)
    b_ols = float(intercept / (a_ols * dt))

    residuals = Y - (intercept + slope * X)
    sigma = float(np.std(residuals) / np.sqrt(dt))

    a = a_ols
    if min_half_life_years is not None:
        a_floor = math.log(2) / min_half_life_years
        a = max(a_ols, a_floor)

    b = b_ols if target_override is None else float(target_override)

    return {
        'type': 'ou',
        'name': col_name,
        'start_value': start_value,
        'a': a,
        'b': b,
        'sigma': sigma,
        # kept for transparency/debugging -- what the raw regression said
        # before any override/floor was applied
        'a_ols': a_ols,
        'b_ols': b_ols,
    }


def _calibrate_jump_params(col_name, series, anchor_series, direction_bias=0.8, min_rate=0.0):
    """Empirical jump-process calibration for a discrete policy rate.

    - p_change: historical monthly probability that the rate actually moved
      (estimated directly from observed frequency of changes, no meeting-date
      modelling involved).
    - step_sizes: empirical (bootstrapped) distribution of historical nonzero
      move sizes, so simulated jumps look like real MPC moves (e.g. clusters
      around 0.25pp / 0.50pp) rather than a smooth Gaussian magnitude.
    - neutral_real_rate / anchor: instead of predicting *when* a move happens,
      the *direction* of a jump (once it occurs) is biased toward closing the
      gap between the current rate and a simple Fisher-style target
      (neutral real rate + simulated inflation). This replaces the old
      levels-based Cholesky correlation with a more defensible causal link:
      NBP reacts to inflation, rather than "NBP and CPI shocks happen to be
      correlated".
    """
    start_value = float(series.iloc[-1])

    diffs = series.diff().dropna()
    changes = diffs[diffs != 0]

    if len(diffs) == 0:
        p_change = 0.0
    else:
        p_change = float(len(changes) / len(diffs))

    if len(changes) > 0:
        step_sizes = changes.abs().values.astype(float)
    else:
        # No historical changes observed in the calibration window; fall back
        # to a conservative single default step. p_change will be 0 so this
        # is never actually sampled in practice.
        step_sizes = np.array([0.0025])

    if anchor_series is not None:
        neutral_real_rate = float((series - anchor_series).mean())
    else:
        neutral_real_rate = 0.0

    return {
        'type': 'jump',
        'name': col_name,
        'start_value': start_value,
        'p_change': p_change,
        'step_sizes': step_sizes,
        'neutral_real_rate': neutral_real_rate,
        'direction_bias': direction_bias,
        'min_rate': min_rate,
    }


def params_calculations(matrix, dt=1/12, cpi_target=CPI_TARGET, cpi_min_half_life_years=CPI_MIN_HALF_LIFE_YEARS):
    """Calibrates each series in `matrix`.

    - Policy-rate columns (name contains 'nbp') are calibrated as an
      empirical jump process anchored to the inflation column.
    - The CPI column keeps the Vasicek/OU diffusion calibration, but its
      long-run mean is anchored to NBP's official inflation target
      (cpi_target) rather than the fragile OLS-implied value, and its
      reversion speed is floored using cpi_min_half_life_years. Set either
      to None to fall back to the raw OLS estimate.
    - Any other column keeps the plain OLS-implied OU calibration.
    """
    anchor_col = None
    for col in matrix.columns:
        if 'cpi' in col.lower():
            anchor_col = matrix[col]
            break

    params_list = []
    for col_name in matrix.columns:
        series = matrix[col_name]

        if _is_policy_rate(col_name):
            params_list.append(_calibrate_jump_params(col_name, series, anchor_col))
        elif 'cpi' in col_name.lower():
            params_list.append(_calibrate_ou_params(
                col_name, series, dt,
                target_override=cpi_target,
                min_half_life_years=cpi_min_half_life_years,
            ))
        else:
            params_list.append(_calibrate_ou_params(col_name, series, dt))

    return params_list


def simulate_paths(stacked_shocks, params_list, horizon, sim_number, dt=1/12):
    """Simulates all paths month by month.

    - 'ou' columns: standard discretized Vasicek step, driven by the
      (correlated) shock stream from calculate_shock/calculate_correlations.
    - 'jump' columns (policy rate): each month, a jump occurs with the
      calibrated historical probability; if it occurs, its magnitude is
      bootstrapped from historical step sizes and its direction is biased
      toward closing the gap to a simulated CPI-anchored target. Otherwise
      the rate stays flat, matching how a policy rate actually behaves
      between decisions. Uses the global numpy random state (seeded upstream
      via np.random.seed for reproducibility), not a private generator.
    """
    num_variables = stacked_shocks.shape[0]

    paths = np.zeros((num_variables, sim_number, horizon))

    for i in range(num_variables):
        paths[i, :, 0] = params_list[i]["start_value"]

    anchor_idx = None
    for i, p in enumerate(params_list):
        if 'cpi' in p['name'].lower():
            anchor_idx = i
            break

    for t in range(1, horizon):

        for i in range(num_variables):
            p = params_list[i]

            if p['type'] == 'ou':
                a = p["a"]
                b = p["b"]
                sigma = p["sigma"]

                drift = a * (b - paths[i, :, t-1]) * dt
                shock = sigma * np.sqrt(dt) * stacked_shocks[i, :, t]

                paths[i, :, t] = paths[i, :, t-1] + drift + shock

            elif p['type'] == 'jump':
                prev = paths[i, :, t-1]

                if anchor_idx is not None:
                    anchor_now = paths[anchor_idx, :, t]
                    target = p['neutral_real_rate'] + anchor_now
                else:
                    target = prev

                gap = target - prev

                jump_occurs = np.random.rand(sim_number) < p['p_change']

                follows_gap = np.random.rand(sim_number) < p['direction_bias']
                gap_sign = np.sign(gap)
                zero_mask = gap_sign == 0
                if zero_mask.any():
                    gap_sign[zero_mask] = np.random.choice([-1.0, 1.0], size=int(zero_mask.sum()))
                random_sign = np.random.choice([-1.0, 1.0], size=sim_number)
                sign = np.where(follows_gap, gap_sign, random_sign)

                magnitude = np.random.choice(p['step_sizes'], size=sim_number, replace=True)

                step = np.where(jump_occurs, sign * magnitude, 0.0)
                new_val = np.maximum(p['min_rate'], prev + step)

                paths[i, :, t] = new_val

            else:
                raise ValueError(f"Unknown params type: {p['type']}")

    return paths