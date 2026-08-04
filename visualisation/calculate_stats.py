import numpy as np

def calculate_statistics(global_matrix, initial_capital, tolerance_percent=5):

    mean_path = np.mean(global_matrix, axis=0)
    worst_path = np.percentile(global_matrix, tolerance_percent, axis=0)
    best_path = np.percentile(global_matrix, 100 - tolerance_percent, axis=0)

    final_values = global_matrix[:, -1]
    
    mean_final_wealth = np.mean(final_values)
    worst_final_wealth = np.percentile(final_values, tolerance_percent)
    best_final_wealth = np.percentile(final_values, 100 - tolerance_percent)

    mean_profit = mean_final_wealth - initial_capital
    worst_profit = worst_final_wealth - initial_capital
    best_profit = best_final_wealth - initial_capital
    
    return {
        "paths": {
            "mean": mean_path,
            "worst": worst_path,
            "best": best_path
        },
        "summary": {
            "mean_wealth": mean_final_wealth,
            "worst_wealth": worst_final_wealth,
            "best_wealth": best_final_wealth,
            "mean_profit": mean_profit,
            "worst_profit": worst_profit,
            "best_profit": best_profit
        }
    }