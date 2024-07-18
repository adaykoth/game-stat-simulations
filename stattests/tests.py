import numpy as np
import scipy.stats
from tqdm import tqdm
from statsmodels.stats.proportion import proportions_ztest

def proportions_test(a,b):

    # first calculate the number of successes and the number of trials for each experiment, then use the proportions_ztest
    # function from statsmodels to calculate the p-value
    n_experiments = a.shape[0]
    n_users = a.shape[1]
    # number of successes consists of all users with one or more purchases
    successes_a = np.sum(a > 0, axis=1)
    successes_b = np.sum(b > 0, axis=1)
    # get the p-value for each experiment
    p_values = np.array([proportions_ztest([successes_a[i], successes_b[i]], [n_users, n_users])[1] for i in range(n_experiments)])
    return p_values

def t_test(a, b):
    """
    Calculates two-sided t-test p-values for multiple experiments
    :param a: np.array shape (n_experiments, n_users), metric values in control group
    :param b: np.array shape (n_experiments, n_users), metric values in treatment group
    :return: np.array shape (n_experiments), two-sided p-values of t-test in all experimetns
    """
    result = list(map(lambda x: scipy.stats.ttest_ind(x[0], x[1]).pvalue, zip(a, b)))
    return np.array(result)

def log_t_test(a, b):
    """
    Calculates two-sided t-test p-values for multiple experiments
    :param a: np.array shape (n_experiments, n_users), metric values in control group
    :param b: np.array shape (n_experiments, n_users), metric values in treatment group
    :return: np.array shape (n_experiments), two-sided p-values of t-test in all experimetns
    """
    log_a = np.log(a + 1)
    log_b = np.log(b + 1)
    result = list(map(lambda x: scipy.stats.ttest_ind(x[0], x[1]).pvalue, zip(log_a, log_b)))
    return np.array(result)

def whale_filtered_t_test(a, b):
    """
    Calculates two-sided t-test p-values for multiple experiments
    :param a: np.array shape (n_experiments, n_users), metric values in control group
    :param b: np.array shape (n_experiments, n_users), metric values in treatment group
    :return: np.array shape (n_experiments), two-sided p-values of t-test in all experimetns
    """
    p_values = []
    for _a,_b in zip(a,b):
        _a = _a[_a < 200]
        _b = _b[_b < 200]
        p_values.append(scipy.stats.ttest_ind(_a, _b).pvalue)
    return np.array(p_values)

def whale_filtered_log_t_test(a, b):
    """
    Calculates two-sided t-test p-values for multiple experiments
    :param a: np.array shape (n_experiments, n_users), metric values in control group
    :param b: np.array shape (n_experiments, n_users), metric values in treatment group
    :return: np.array shape (n_experiments), two-sided p-values of t-test in all experimetns
    """
    p_values = []
    for _a,_b in zip(a,b):
        _a = _a[_a < 250]
        _b = _b[_b < 250]
        _a = np.log(_a + 1)
        _b = np.log(_b + 1)
        p_values.append(scipy.stats.ttest_ind(_a, _b).pvalue)
    return np.array(p_values)

def cuped_t_test(a, b, a_pre, b_pre):
    def calculate_theta(pre_control_data, pre_variant_data, post_control_data, post_variant_data):
        n_experiments, n_users = pre_control_data.shape
        theta = np.empty(n_experiments)

        for i in range(n_experiments):
            # Combine pre-experiment data from control and variant groups
            combined_pre_data = np.concatenate((pre_control_data[i], pre_variant_data[i]))
            combined_post_data = np.concatenate((post_control_data[i], post_variant_data[i]))

            # Calculate means
            pre_mean = np.mean(combined_pre_data)
            post_mean = np.mean(combined_post_data)

            # Calculate covariance and variance
            cov = np.cov(combined_pre_data, combined_post_data)[0, 1]
            var_pre = np.var(combined_pre_data)

            # Calculate theta
            theta[i] = cov / var_pre

        return theta
    def apply_cuped(pre_data, post_data, theta):
        n_experiments, n_users = pre_data.shape
        adjusted_data = np.empty_like(post_data)

        for i in range(n_experiments):
            pre_mean = np.mean(pre_data[i])
            post_mean = np.mean(post_data[i])
            cov = np.cov(pre_data[i], post_data[i])[0, 1]
            var_pre = np.var(pre_data[i])
            theta[i] = cov / var_pre
            adjusted_data[i] = post_data[i] - theta[i] * (pre_data[i] - pre_mean) + post_mean

        return adjusted_data

    theta = calculate_theta(a_pre, b_pre, a, b)
    a = apply_cuped(a_pre, a, theta)
    b = apply_cuped(b_pre, b, theta)

    result = list(map(lambda x: scipy.stats.ttest_ind(x[0], x[1]).pvalue, zip(a, b)))
    return np.array(result)

def cuped_log_t_test(a, b, a_pre, b_pre):
    def calculate_theta(pre_control_data, pre_variant_data, post_control_data, post_variant_data):
        n_experiments, n_users = pre_control_data.shape
        theta = np.empty(n_experiments)

        for i in range(n_experiments):
            # Combine pre-experiment data from control and variant groups
            combined_pre_data = np.concatenate((pre_control_data[i], pre_variant_data[i]))
            combined_post_data = np.concatenate((post_control_data[i], post_variant_data[i]))

            # Calculate means
            pre_mean = np.mean(combined_pre_data)
            post_mean = np.mean(combined_post_data)

            # Calculate covariance and variance
            cov = np.cov(combined_pre_data, combined_post_data)[0, 1]
            var_pre = np.var(combined_pre_data)

            # Calculate theta
            theta[i] = cov / var_pre

        return theta
    def apply_cuped(pre_data, post_data, theta):
        n_experiments, n_users = pre_data.shape
        adjusted_data = np.empty_like(post_data)

        for i in range(n_experiments):
            pre_mean = np.mean(pre_data[i])
            post_mean = np.mean(post_data[i])
            cov = np.cov(pre_data[i], post_data[i])[0, 1]
            var_pre = np.var(pre_data[i])
            theta[i] = cov / var_pre
            adjusted_data[i] = post_data[i] - theta[i] * (pre_data[i] - pre_mean) + post_mean

        return adjusted_data

    a = np.log(a + 1)
    b = np.log(b + 1)
    a_pre = np.log(a_pre + 1)
    b_pre = np.log(b_pre + 1)
    theta = calculate_theta(a_pre, b_pre, a, b)
    a = apply_cuped(a_pre, a, theta)
    b = apply_cuped(b_pre, b, theta)


    result = list(map(lambda x: scipy.stats.ttest_ind(x[0], x[1]).pvalue, zip(a, b)))
    return np.array(result)

def mannwhitney(a, b):
    """
    Calculates two-sided t-test p-values for multiple experiments
    :param a: np.array shape (n_experiments, n_users), metric values in control group
    :param b: np.array shape (n_experiments, n_users), metric values in treatment group
    :return: np.array shape (n_experiments), two-sided p-values of Mann-Whitney test in all experimetns
    """
    result = list(map(lambda x: scipy.stats.mannwhitneyu(x[0], x[1], alternative='two-sided').pvalue, zip(a, b)))
    return np.array(result)


def get_smoothed_ctrs(clicks_0, views_0, clicks_1, views_1, smothing_factor=200.):
    """
    Calculates smoothed ctr for every user in every experiment both in treatment and control groups
    Smoothed_ctr = (user_clicks + smothing_factor * global_ctr) / (user_views + smothing_factor)
    :param clicks_0: np.array shape (n_experiments, n_users), clicks of every user from control group in every experiment
    :param views_0: np.array shape (n_experiments, n_users), views of every user from control group in every experiment
    :param clicks_1: np.array shape (n_experiments, n_users), clicks of every user from treatment group in every experiment
    :param views_1: np.array shape (n_experiments, n_users), views of every user from treatment group in every experiment
    :param smothing_factor: float
    :return: (np.array, np.array) shape (n_experiments, n_users), smoothed ctrs for every user in every experiment
    """
    global_ctr = (np.sum(clicks_0, axis=1) / np.sum(views_0, axis=1)).reshape(-1, 1)
    ctrs_0 = (clicks_0 + smothing_factor * global_ctr) / (views_0 + smothing_factor)
    ctrs_1 = (clicks_1 + smothing_factor * global_ctr) / (views_1 + smothing_factor)
    return ctrs_0, ctrs_1


def bootstrap_p_values(control: np.ndarray, variant: np.ndarray, n_iterations: int = 500) -> np.ndarray:
    """
    Calculate p-values from bootstrapping for control and variant data.

    Parameters:
    control (np.ndarray): 2D array of control data with shape (experiments, users).
    variant (np.ndarray): 2D array of variant data with shape (experiments, users).
    n_iterations (int): Number of bootstrap iterations.

    Returns:
    np.ndarray: Array of p-values for each experiment.
    """

    combine_data = np.concatenate([control, variant], axis=1)

    n_experiments, n_users = combine_data.shape
    p_values = np.empty(n_experiments)
    for i in tqdm(range(n_experiments)):
        control_data = control[i]
        variant_data = variant[i]
        observed_diff = np.mean(variant_data) - np.mean(control_data)
        data = combine_data[i]
        # Generate bootstrap samples and calculate statistics
        bootstrap_means_control = np.empty(n_iterations)
        bootstrap_means_variant = np.empty(n_iterations)

        for j in range(n_iterations):
            bootstrap_sample = np.random.choice(data, size=n_users, replace=True)

            bootstrap_means_control[j] = np.mean(bootstrap_sample[len(control_data):])
            bootstrap_means_variant[j] = np.mean(bootstrap_sample[:len(control_data)])

        # Calculate the bootstrap distribution of the difference in means
        bootstrap_diff = bootstrap_means_variant - bootstrap_means_control

        # Calculate the p-value for one-tailed test (variant > control)

        p_value = np.mean(bootstrap_diff >= observed_diff)
        p_values[i] = p_value

    return p_values