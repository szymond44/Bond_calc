import numpy as np
import math

def capital_calculator(
    macro_sim_matrix,
    belka_tax_rate,
    initial_capital,
    capitalisation_period,
    timeframe_months,
    num_sim,
    reinvest,
    does_capitalise,
    strategy_horizon,
    global_start_month=0,
    global_capital_matrix=None,
    early_buyout_penalty=0.0,
):
    """Computes capital growth for a single bond within a strategy.

    strategy_horizon is the number of columns the (possibly shared) capital
    matrix is allocated with. Callers wanting to cap a strategy at a
    user-chosen investment horizon that's shorter than the bond sequence's
    full natural length should simply pass that shorter horizon here (via
    simulate_strategy) instead of the full sequence length -- this function
    then never computes or stores capital past that many columns.

    early_buyout_penalty is this bond's early-redemption penalty rate (from
    bonds_config.json). It's applied automatically, but only if this bond's
    own natural completion (global_start_month + timeframe_months) falls
    beyond strategy_horizon -- i.e. only if the matrix actually gets cut off
    while this bond is still running, meaning the investor exited before
    this bond matured. If the bond completes exactly at or before
    strategy_horizon, no penalty is applied, since that's a natural
    redemption, not an early one. When no cutoff is in play (strategy_horizon
    covers the whole sequence, as before this feature existed), this is
    always the case and the penalty never triggers.
    """
    periods_num = math.ceil(timeframe_months / capitalisation_period)
    use_global = global_capital_matrix is not None

    if use_global:
        capital_matrix = global_capital_matrix
    else:
        capital_matrix = np.zeros((num_sim, strategy_horizon))

    matrix_width = capital_matrix.shape[1]

    if np.isscalar(initial_capital):
        current_capital = np.full(num_sim, float(initial_capital))
    else:
        current_capital = np.asarray(initial_capital, dtype=float).copy()
        if current_capital.shape != (num_sim,):
            raise ValueError("initial_capital array must have shape (num_sim,)")

    non_working_capital = np.zeros(num_sim)

    bond_end_global = global_start_month + timeframe_months
    is_early_redemption = bond_end_global > matrix_width

    for period in range(periods_num):
        start_month = period * capitalisation_period
        end_month = min((period + 1) * capitalisation_period, timeframe_months)

        g0 = global_start_month + start_month
        if g0 >= matrix_width:
            break
        g1 = min(global_start_month + end_month, matrix_width)

        fraction_of_year = (end_month - start_month) / 12.0
        rate_for_period = macro_sim_matrix[:, start_month]
        gross_profit = current_capital * rate_for_period * fraction_of_year

        if does_capitalise:
            current_capital = current_capital + gross_profit
        else:
            net_profit = gross_profit * (1 - belka_tax_rate)
            if reinvest:
                current_capital = current_capital + net_profit
            else:
                non_working_capital = non_working_capital + net_profit

        total_wealth = current_capital + non_working_capital
        capital_matrix[:, g0:g1] = total_wealth[:, np.newaxis]

        if (not use_global) and timeframe_months < strategy_horizon:
            final_capital = current_capital + non_working_capital
            capital_matrix[:, timeframe_months:strategy_horizon] = final_capital[:, np.newaxis]

    if is_early_redemption and early_buyout_penalty > 0 and matrix_width > 0:
        last_col = matrix_width - 1
        capital_matrix[:, last_col] = capital_matrix[:, last_col] * (1 - early_buyout_penalty)

    return capital_matrix, current_capital, non_working_capital
