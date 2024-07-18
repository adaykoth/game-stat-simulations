from typing import Tuple

import numpy as np


def generate_data(
                  N: int = 50000,
                  NN: int = 2000,
                  purchases_mean: float = 0.02475,
                  purchases_var: float = 0.3420,
                  purchase_rev_log_mean: float = 0.1909,
                  purchase_rev_log_std: float = 1.7588,
                  uplift_purchase_rate: float = 0.1,
                    uplift_purchase_amount: float = 0.0) -> Tuple[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:

    r_0 = (purchases_mean ** 2 / (purchases_var - purchases_mean))
    p_0 = purchases_mean / purchases_var

    purchase_uplifted = purchases_mean * (1 + uplift_purchase_rate)
    r_1 = (purchase_uplifted ** 2 / (purchases_var - purchase_uplifted))
    p_1 = purchase_uplifted / purchases_var

    purchases_0 = np.random.negative_binomial(r_0, p_0, NN*N).reshape(NN, N)
    purchases_1 = np.random.negative_binomial(r_1, p_1, NN*N).reshape(NN, N)


    # views are always positive, abs is fixing numerical issues with high skewness
    purchases_0 = np.absolute(purchases_0)
    purchases_1 = np.absolute(purchases_1)

    rev_log_mean_0 = purchase_rev_log_mean
    rev_log_mean_1 = purchase_rev_log_mean + np.log(1+uplift_purchase_amount)

    # Flatten purchases array to generate revenue and then reshape
    purchase_revenue_0 = np.random.lognormal(rev_log_mean_0, purchase_rev_log_std, np.sum(purchases_0))
    purchase_revenue_1 = np.random.lognormal(rev_log_mean_1, purchase_rev_log_std, np.sum(purchases_1))

    # Assign purchase revenues to users
    user_revenue_list_0 = np.split(purchase_revenue_0, np.cumsum(purchases_0.flatten())[:-1])
    user_revenue_list_1 = np.split(purchase_revenue_1, np.cumsum(purchases_1.flatten())[:-1])

    # Calculate total revenue per user for each experiment
    total_revenue_0 = np.array(
        [np.sum(revenues) if len(revenues) > 0 else 0 for revenues in user_revenue_list_0]).reshape(NN, N)
    total_revenue_1 = np.array(
        [np.sum(revenues) if len(revenues) > 0 else 0 for revenues in user_revenue_list_1]).reshape(NN, N)

    return ((purchases_0.astype(np.float64), total_revenue_0.astype(np.float64)),
            (purchases_1.astype(np.float64), total_revenue_1.astype(np.float64)))