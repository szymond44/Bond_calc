import numpy as np
import math


def interest_rate_calculator(timeframe_months, bonus_length, bonus_rate, margin, capitalisation_period, index_paths, num_sim, global_start_month=0): 
    periods_num = math.ceil(timeframe_months / capitalisation_period)
    macro_sim_matrix = np.zeros((num_sim, timeframe_months)) 
    path_len = index_paths.shape[1]
    
    for period in range(periods_num):
        start_month = period * capitalisation_period
        end_month = min((period + 1) * capitalisation_period, timeframe_months)

        take_macro_matrix = global_start_month + start_month
        if take_macro_matrix >= path_len:
            raise IndexError(
                f"Macro lock month {take_macro_matrix} >= path length {path_len}. "
                "Increase T in path simulation or shorten strategy / global_start_month."
            )
        period_locked_ir = index_paths[:, [take_macro_matrix]]

        macro_sim_matrix[:, start_month:end_month] = np.maximum(0, period_locked_ir + margin)

        if bonus_rate > 0 and start_month < bonus_length:
            bonus_end = min(bonus_length, end_month)
            macro_sim_matrix[:, start_month:bonus_end] = bonus_rate
            
    return macro_sim_matrix, timeframe_months