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


def params_calculations(matrix, dt=1/12):
    params_list = []

    for col_name in matrix.columns:

        s = matrix[col_name]

        start_value = float(s.iloc[-1])

        X = s[:-1].values
        Y = np.diff(s.values)

        slope, intercept, _, _, _ = linregress(X, Y)
        a = float(-slope / dt)
        b = float(intercept / (a * dt))

        residuals = Y - (intercept + slope * X)
        sigma = float(np.std(residuals) / np.sqrt(dt))

        index_name = col_name

        params_list.append({
            'name': index_name,
            'start_value': start_value,
            'a': a,
            'b': b,
            'sigma': sigma
        })

    return params_list


def simulate_paths(stacked_shocks, params_list, horizon, sim_number, dt=1/12):

    num_variables = stacked_shocks.shape[0]

    paths = np.zeros((num_variables, sim_number, horizon))

    for i in range(num_variables):
        paths[i, :, 0] = params_list[i]["start_value"]

    for t in range(1, horizon):

        for i in range(num_variables):

            a = params_list[i]["a"]
            b = params_list[i]["b"]
            sigma = params_list[i]["sigma"]

            drift = a * (b - paths[i, :, t-1]) * dt
            shock = sigma * np.sqrt(dt) * stacked_shocks[i, :, t]

            paths[i, :, t] = paths[i, :, t-1] + drift + shock

    return paths
