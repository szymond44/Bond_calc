from simulation.capital_calculator import capital_calculator
from engine.strategy_horizon import calculate_strategy_horizon
from simulation.get_bond_rates import get_bond_rates

def simulate_strategy(strategy, initial_capital, paths_mapping, num_sim, belka_tax_rate=0.19, reinvest=True, max_horizon_months=None):
    """Runs a sequence of bonds one after another, feeding each bond's ending
    capital into the next.

    max_horizon_months: optional cap, in months, on how long the investor
    actually stays invested. If None (default), the strategy runs for the
    full natural length of its bond sequence, exactly as before this
    parameter existed. If given and shorter than the sequence's full length,
    the strategy is cut at that point: bonds starting after the cutoff are
    never simulated, and whichever bond is still running at the cutoff has
    its early_buyout_penalty (from bonds_config.json) applied to the capital
    at that point, since the investor would be redeeming it early. If the
    cutoff lands exactly on the boundary between two bonds, no penalty
    applies, since that bond matured naturally right at that point.

    Returns (global_matrix, current_capital, effective_horizon, penalty_info).
    effective_horizon is the number of months actually simulated (equal to
    the full sequence length unless cut short). penalty_info is None unless
    a penalty was applied, in which case it's {"bond": name, "rate": rate}.
    """

    full_horizon = calculate_strategy_horizon(strategy)
    effective_horizon = full_horizon if max_horizon_months is None else min(full_horizon, max_horizon_months)

    current_month = 0
    current_capital = initial_capital
    global_matrix = None
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

    return global_matrix, current_capital, effective_horizon, penalty_info
