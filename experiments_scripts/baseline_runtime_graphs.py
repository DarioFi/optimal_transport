import matplotlib.pyplot as plt
from opt_trans.data_handling.experiment_data import Database
from collections import defaultdict

folder = '../run_logs/baseline_runtime_performance/'
db = Database.populate_from_folder(folder)
data = db.index_on('formulation', 'instance_arguments')


def plot_data(data, formulation_name, ax, x_shift):
    n_values, times, stds = [], [], []
    for key, runs in data.items():
        if key[1] != formulation_name:
            continue

        n = list(key[3])[0][1]
        good = [r for r in runs if r.results['termination_condition'] == 'optimal']
        tot = len(runs)
        succ = len(good)

        # compute mean/std or use timeout
        if succ > 0:
            mean_t = sum(r.results['time'] for r in good) / succ
            std_t = (sum((r.results['time'] - mean_t) ** 2 for r in good) / succ) ** 0.5
        else:
            mean_t, std_t = 300, 0

        if succ != 0:
            n_values.append(n)
            times.append(mean_t)
            stds.append(std_t)

        # annotate only if not all succeeded
        if succ != tot:
            # shift the annotation left/right to avoid overlap
            if formulation_name == 'dbt_alpha_0' and n == 5:
                xytext = (x_shift, 30)
            else:
                xytext = (x_shift, 10)

            if formulation_name == 'dbt_alpha_0_NO_BO' and n == 7:
                xytext = (x_shift, -10)


            match formulation_name:
                case 'dbt_alpha_0':
                    color = 'C0'
                case 'dbtq':
                    color = 'C1'
                case 'dbt_alpha_0_NO_BO':
                    color = 'C2'
            ax.annotate(f"{succ}/{tot} converged",
                        (n, mean_t),
                        textcoords="offset points",
                        xytext=xytext,
                        ha='center',
                        va='bottom',
                        fontsize=12,
                        color=color,
                        bbox=dict(boxstyle="round,pad=0.3",
                                  fc="white", alpha=0.7,
                                  ec=color)
                        )

    # sort by n

    goodnames = {
        'dbt_alpha_0': 'DBT with BO',
        'dbtq': 'DBTQ',
        'dbt_alpha_0_NO_BO': 'DBT'
    }

    xs, ys, es = zip(*sorted(zip(n_values, times, stds)))
    ax.errorbar(xs, ys, yerr=es,
                label=goodnames[formulation_name],
                marker='^', capsize=5)


def plot_all_data(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    # for alpha=0, shift left by -10 points; for dbtq, shift right +10
    plot_data(data, 'dbt_alpha_0', ax, x_shift=-10)
    plot_data(data, 'dbtq', ax, x_shift=+10)
    plot_data(data, 'dbt_alpha_0_NO_BO', ax, x_shift=0)

    ax.set_xlabel('n', fontsize=14)
    ax.set_ylabel('time (s)', fontsize=14)
    ax.set_xticks([4, 5, 6, 7])
    ax.set_ylim(0, 310)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=12)
    fig.tight_layout()
    plt.show()
    fig.savefig('baseline_runtime_performance.png', dpi=300)


plot_all_data(data)
