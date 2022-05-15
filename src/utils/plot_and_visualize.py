import matplotlib.pyplot as plt
import wandb

plt.style.use('seaborn-whitegrid')


def plot_confidence_interval(ax, x_axis, mean_value, bottom_value, top_value, color, label):
    ax.plot(x_axis, mean_value, 'o-', color=color, label=label, linewidth=1, markersize=3)
    ax.plot([x_axis, x_axis], [bottom_value, top_value], color=color, linewidth=1)


def plot_graph_amplification_bias(fairness_measure, suffix, x_axis, use_wandb, methods):
    fig = plt.figure()
    ax = plt.axes()
    ax.plot([i - 0.1 for i in x_axis], methods['smoothed_amplification'], 'o-', color='tab:blue',
            label='smoothed', linewidth=1, markersize=3)
    ax.plot(x_axis, methods['simple_bayesian_amplification'], 'o-', color='tab:green',
            label='bayesian', linewidth=1, markersize=3)
    ax.plot([i + 0.1 for i in x_axis], methods['bootstrap_amplification'], 'o-', color='tab:blue',
            label='bootstrap', linewidth=1, markersize=3)

    plt.xlabel(suffix + "Epoch")
    plt.ylabel(f"{fairness_measure} bias amplification")
    plt.legend()

    if use_wandb:
        wandb.log({f"{suffix + fairness_measure} bais amplification": wandb.Image(plt)})
    plt.savefig(f"../plots/{suffix + fairness_measure}_bias_amplification")


def plot_graph(fairness_measure, suffix, x_axis, use_wandb, methods):
    def _temp(info):
        if len(info[0]) == 2:
            return [i[0] for i in info], [i[1][0] for i in info], [i[1][1] for i in info]
        else:
            return info, [0.0 for _ in info], [0.0 for _ in info]

    fig = plt.figure()
    ax = plt.axes()

    smoothed_value, smoothed_error_bottom, smoothed_error_top = _temp(methods['smoothed'])
    simple_bayesian_value, simple_bayesian_error_bottom, simple_bayesian_error_top = _temp(methods['simple_bayesian'])
    bootstrap_value, bootstrap_error_bottom, bootstrap_error_top = _temp(methods['bootstrap'])

    plot_confidence_interval(ax, [i - 0.1 for i in x_axis], smoothed_value, smoothed_error_bottom, smoothed_error_top,
                             'tab:blue', 'smoothed')
    plot_confidence_interval(ax, x_axis, simple_bayesian_value, simple_bayesian_error_bottom, simple_bayesian_error_top,
                             'tab:green', 'bayesian')
    plot_confidence_interval(ax, [i + 0.1 for i in x_axis], bootstrap_value, bootstrap_error_bottom,
                             bootstrap_error_top, 'tab:orange',
                             'bootstrap')

    plt.xlabel(suffix + "Epoch")
    plt.ylabel(f"{fairness_measure}")
    plt.legend()

    if use_wandb:
        wandb.log({f"{suffix + fairness_measure}": wandb.Image(plt)})
    plt.savefig(f"../plots/{suffix + fairness_measure}")


def plot_eps_fairness_metric(eps_fairness, suffix="Train ", use_wandb=False):
    fairness_modes = eps_fairness[0].keys()
    for fairness_measure in fairness_modes:
        smoothed = []
        simple_bayesian = []
        bootstrap = []

        smoothed_amplification = []
        simple_bayesian_amplification = []
        bootstrap_amplification = []

        for eps in eps_fairness:  # for each epoch
            smoothed.append(eps[fairness_measure].intersectional_smoothed)
            simple_bayesian.append(eps[fairness_measure].intersectional_simple_bayesian)
            bootstrap.append(eps[fairness_measure].intersectional_bootstrap)
            # add amplifcation too
            smoothed_amplification.append(eps[fairness_measure].intersectional_smoothed_bias_amplification)
            simple_bayesian_amplification.append(
                eps[fairness_measure].intersectional_simple_bayesian_bias_amplification)
            bootstrap_amplification.append(eps[fairness_measure].intersectional_bootstrap_bias_amplification)

        x_axis = [i + 1 for i in range(len(eps_fairness))]
        methods = {
            'smoothed': smoothed,
            'simple_bayesian': simple_bayesian,
            'bootstrap': bootstrap
        }

        plot_graph(fairness_measure, suffix, x_axis, use_wandb, methods)
        if fairness_measure == 'demographic_parity':
            methods['smoothed_amplification'] = smoothed_amplification
            methods['simple_bayesian_amplification'] = simple_bayesian_amplification
            methods['bootstrap_amplification'] = bootstrap_amplification

            plot_graph_amplification_bias(fairness_measure, suffix, x_axis, use_wandb, methods)
