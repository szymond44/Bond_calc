import numpy as np

from simulation.capital_calculator import capital_calculator
from engine.strategy_horizon import calculate_strategy_horizon
from simulation.get_bond_rates import get_bond_rates

def simulate_strategy(strategy, initial_capital, paths_mapping, num_sim, belka_tax_rate=0.19, reinvest=True, max_horizon_months=None, dca_amount=0.0, dca_duration_months=0):
    full_horizon = calculate_strategy_horizon(strategy)
    effective_horizon = full_horizon if max_horizon_months is None else min(full_horizon, max_horizon_months)

    master_global_matrix = np.zeros((num_sim, effective_horizon))
    
    actual_dca_duration = min(dca_duration_months, effective_horizon - 1) if dca_duration_months > 0 else 0
    total_invested = 0.0
    
    for t in range(actual_dca_duration + 1):
        if t == 0:
            start_capital = initial_capital
            if dca_amount > 0 and dca_duration_months >= 0:
                start_capital += dca_amount
        else:
            if dca_amount <= 0:
                break
            start_capital = dca_amount
            
        total_invested += start_capital
            
        if start_capital <= 0:
            continue

        current_month = t
        current_capital = np.full(num_sim, float(start_capital))
        global_matrix = np.zeros((num_sim, effective_horizon))
        
        penalty_info = None

        for bond in strategy:
            if current_month >= effective_horizon:
                break

            rates_matrix, timeframe = get_bond_rates(
                bond_dict=bond,
                paths_map=paths_mapping,
                num_sim=num_sim,
                global_start=current_month
            )

            bond_penalty_rate = bond.get("early_buyout_penalty", 0.0)

            global_matrix, cap, nw_cap = capital_calculator(
                macro_sim_matrix=rates_matrix,
                belka_tax_rate=belka_tax_rate,
                initial_capital=current_capital,
                capitalisation_period=bond["capitalisation_period"],
                timeframe_months=timeframe,
                num_sim=num_sim,
                reinvest=reinvest,
                does_capitalise=bond["does_capitalise"],
                strategy_horizon=effective_horizon,
                global_start_month=current_month,
                global_capital_matrix=global_matrix,
                early_buyout_penalty=bond_penalty_rate,
            )

            bond_end_global = current_month + timeframe
            if bond_end_global > effective_horizon and bond_penalty_rate > 0:
                penalty_info = {"bond": bond.get("name"), "rate": bond_penalty_rate}

            current_capital = cap + nw_cap
            current_month = bond_end_global

        master_global_matrix += global_matrix

    return master_global_matrix, total_invested, effective_horizon, penalty_info