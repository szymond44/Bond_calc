from capital_calculator import capital_calculator
from engine.strategy_horion import calculate_strategy_horizon
from get_bond_rates import get_bond_rates

def simulate_strategy(strategy, initial_capital, paths_mapping, num_sim, belka_tax_rate=0.19, reinvest=True):

    strategy_horizon = calculate_strategy_horizon(strategy)

    current_month = 0
    current_capital = initial_capital
    global_matrix = None  

    for bond in strategy:

        rates_matrix, timeframe = get_bond_rates(
            bond_dict=bond, 
            paths_map=paths_mapping, 
            num_sim=num_sim, 
            global_start=current_month
        )

        global_matrix, cap, nw_cap = capital_calculator(
            macro_sim_matrix=rates_matrix,
            belka_tax_rate=belka_tax_rate,
            initial_capital=current_capital,     
            capitalisation_period=bond["capitalisation_period"],
            timeframe_months=timeframe,
            num_sim=num_sim,
            reinvest=reinvest,
            does_capitalise=bond["does_capitalise"],
            strategy_horizon=strategy_horizon,
            global_start_month=current_month,    
            global_capital_matrix=global_matrix 
        )
        
        
        current_capital = cap + nw_cap
        current_month += timeframe
        
    return global_matrix, current_capital