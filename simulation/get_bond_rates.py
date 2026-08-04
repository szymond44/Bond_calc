from interest_rate_calculator import interest_rate_calculator

def get_bond_rates(bond_dict, paths_map, num_sim, global_start=0):

    return interest_rate_calculator(
        timeframe_months=bond_dict["timeframe_months"],
        bonus_length=bond_dict["bonus_length"],
        bonus_rate=bond_dict["bonus_rate"],
        margin=bond_dict["margin"],
        capitalisation_period=bond_dict["capitalisation_period"],
        index_paths=paths_map[bond_dict["index_type"]],
        num_sim=num_sim,
        global_start_month=global_start
    )