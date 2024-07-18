from typing import Tuple, Dict


import numpy as np
import scipy.stats
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

colors = sns.color_palette("deep")

codenames2titles = {
    'ttest_successes_count': ('T-test, clicks', colors[0]),
    'mannwhitney_successes_count': ('Mann-Whitney test, clicks', colors[1]),
    'delta': ('Delta-method, global CTR', colors[2]),
    'bootstrap': ('Bootstrap, global CTR', colors[3]),
    'permutation_test': ('Permutation, global CTR', colors[8]),
    'linearization': ('Linearization, clicks', colors[4]),
    'buckets_ctrs': ('Bucketization, global CTR', colors[5]),
    't_test_ctrs': ('T-test, user CTR', colors[6]),
    'mw_ctrs': ('MW, user CTR', colors[7]),
    'weighted_bootstrap': ('Weighted bootstrap, global CTR', colors[7]),
    'weighted_linearization': ('Weighted linearization, global CTR', colors[4]),
    'weighted_buckets': ('Weighted bucketization, global CTR', colors[5]),
    'weighted_t_test_ctrs': ('Weighted t-test, user CTR', colors[6]),
    'weighted_sqr_bootstrap': ('Weighted sqr bootstrap, global CTR', colors[3]),
    'weighted_sqr_linearization': ('Weighted sqr linearization, global CTR', colors[4]),
    'weighted_sqr_buckets': ('Weighted sqr bucketization, global CTR', colors[5]),
    'weighted_sqr_t_test_ctrs': ('Weighted sqr t-test, user CTR', colors[6]),
    'ttest_smoothed': ('T-test, smoothed user CTR', colors[0]),
    'mw_smoothed': ('MW, smoothed user CTR', colors[1]),
    'binomial_test': ('Binomial z-test', colors[7]),
}

cdf_h1_title = 'Simulated p-value CDFs under H1 (Sensitivity)'
cdf_h0_title = 'Simulated p-value CDFs under H0 (FPR)'



def plot_cdf(data: np.ndarray, label: str, ax: Axes, color: str = colors[0], linewidth: float = 3):
    sorted_data = np.sort(data)
    position = scipy.stats.rankdata(sorted_data, method='ordinal')
    cdf = position / data.shape[0]

    sorted_data = np.hstack((sorted_data, 1))
    cdf = np.hstack((cdf, 1))

    return ax.plot(sorted_data, cdf, color=color, linestyle='solid', label=label, linewidth=linewidth)


def plot_summary(dict2plot: Dict[str, Tuple[np.ndarray, np.ndarray, str]],
                 purchases: np.ndarray,
                 purchase_revenue: np.ndarray,
                 control_mean,
                 variant_mean):
    fig = plt.figure(constrained_layout=False, figsize=(4 * 3, 3 * 3), dpi=100)
    gs = fig.add_gridspec(3, 3)
    ax_h1 = fig.add_subplot(gs[:2, :2])
    ax_h0 = fig.add_subplot(gs[0, 2])
    ax_views = fig.add_subplot(gs[1, 2])
    ax_clicks = fig.add_subplot(gs[2, 2])
    ax_powers = fig.add_subplot(gs[2, :2])

    fig.subplots_adjust(left=0.2, wspace=0.3, hspace=0.4)

    ax_h1.plot(np.linspace(0, 1, 10000), np.linspace(0, 1, 10000), 'k', alpha=0.1)
    ax_h0.plot(np.linspace(0, 1, 10000), np.linspace(0, 1, 10000), 'k', alpha=0.1)

    # ax_h1.set_xlabel('p-value')
    # ax_h0.set_xlabel('p-value')
    ax_h1.set_title(cdf_h1_title)
    ax_h0.set_title(cdf_h0_title)

    # ax_h1.set_ylabel('Sensitivity')
    # ax_h0.set_ylabel('FPR')

    ax_h1.axvline(0.05, color='k', alpha=0.5)
    # ax_h1.set_xticks(list(ax_h1.get_xticks()) + [0.05])

    for title, (ab_pvals, aa_pvals, color) in dict2plot.items():
        plot_cdf(ab_pvals, title, ax_h1, color, linewidth=3)
        plot_cdf(aa_pvals, title, ax_h0, color, linewidth=1.5)

    ax_powers.set_title('Test Power')
    tests_powers = []
    tests_labels = []
    tests_colours = []
    for title, (ab_pvals, _, color) in dict2plot.items():
        tests_labels.append(title)
        tests_colours.append(color)
        tests_powers.append(np.mean(ab_pvals < 0.05))
    ax_powers.barh(np.array(tests_labels), np.array(tests_powers), color=np.array(tests_colours))

    sns.histplot(purchases.ravel(),
                 binrange=(0, 150),
                 binwidth=10,
                 ax=ax_views,
                 kde=False)
    ax_views.set_xlim((0, 100))
    ax_views.set_yscale('log')
    purchases_99_percentile = np.percentile(purchases.ravel(), 99.5)
    ax_views.set_title(f'purchases, 99.5%-ile = {purchases_99_percentile:<7.1f}')

    sns.histplot(purchase_revenue.ravel(),
                 binrange=(0, 500),
                 binwidth=20,
                 ax=ax_clicks,
                 kde=False)
    ax_clicks.set_xlim((0, 500))
    ax_clicks.set_yscale('log')
    purchase_revenue_99_percentile = np.percentile(purchase_revenue.ravel(), 99.5)
    ax_clicks.set_title(f'purchase revenue 99.5%-ile = {purchase_revenue_99_percentile:2.3f}')
    # print the mean of control and variant mean at the bottom
    ax_clicks.text(0.5, 0.5, f'Control mean: {control_mean:.3f}\nVariant mean: {variant_mean:.3f}',
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax_clicks.transAxes,
                   fontsize=12)
    return fig
