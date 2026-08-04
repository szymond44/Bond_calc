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
):
    periods_num = math.ceil(timeframe_months / capitalisation_period)
    use_global = global_capital_matrix is not None

    if use_global:
        capital_matrix = global_capital_matrix
    else:
        capital_matrix = np.zeros((num_sim, strategy_horizon))

    if np.isscalar(initial_capital):
        current_capital = np.full(num_sim, float(initial_capital))
    else:
        current_capital = np.asarray(initial_capital, dtype=float).copy()
        if current_capital.shape != (num_sim,):
            raise ValueError("initial_capital array must have shape (num_sim,)")

    non_working_capital = np.zeros(num_sim)

    for period in range(periods_num):
        start_month = period * capitalisation_period
        end_month = min((period + 1) * capitalisation_period, timeframe_months)
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
        g0 = global_start_month + start_month
        g1 = global_start_month + end_month
        capital_matrix[:, g0:g1] = total_wealth[:, np.newaxis]

        if (not use_global) and timeframe_months < strategy_horizon:
            final_capital = current_capital + non_working_capital
            capital_matrix[:, timeframe_months:strategy_horizon] = final_capital[:, np.newaxis]

    return capital_matrix, current_capital, non_working_capital