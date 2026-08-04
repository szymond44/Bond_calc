def calculate_strategy_horizon(strategy):
    return sum(bond['timeframe_months'] for bond in strategy)
