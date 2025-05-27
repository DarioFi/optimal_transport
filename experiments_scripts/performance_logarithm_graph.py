import matplotlib.pyplot as plt
import numpy as np
from opt_trans.data_handling.experiment_data import Database, C

# 1) Load your data and index by the formulation arguments
folder = '../run_logs/baseline_runtime_performance_logarithm/'
db     = Database.populate_from_folder(folder)
data   = db.index_on('instance_arguments', C('formulation_arguments//log_multiplier'))

print(data.keys())

for k in data.keys():
    """('instance_arguments', frozenset({('n', 7)}), 'formulation_arguments//log_multiplier', 20)
-----
('instance_arguments', frozenset({('n', 7)}), 'formulation_arguments//log_multiplier', 10)
-----
('instance_arguments', frozenset({('n', 6)}), 'formulation_arguments//log_multiplier', 100)
-----
('instance_arguments', frozenset({('n', 5)}), 'formulation_arguments//log_multiplier', 20)
-----
('instance_arguments', frozenset({('n', 6)}), 'formulation_arguments//log_multiplier', 10)
-----
('instance_arguments', frozenset({('n', 4)}), 'formulation_arguments//log_multiplier', 100)
-----
('instance_arguments', frozenset({('n', 7)}), 'formulation_arguments//log_multiplier', 40)
-----
('instance_arguments', frozenset({('n', 6)}), 'formulation_arguments//log_multiplier', 40)
-----
('instance_arguments', frozenset({('n', 4)}), 'formulation_arguments//log_multiplier', 40)
-----
('instance_arguments', frozenset({('n', 7)}), 'formulation_arguments//log_multiplier', 100)
-----
('instance_arguments', frozenset({('n', 5)}), 'formulation_arguments//log_multiplier', 10)
-----
('instance_arguments', frozenset({('n', 4)}), 'formulation_arguments//log_multiplier', 10)
-----
('instance_arguments', frozenset({('n', 6)}), 'formulation_arguments//log_multiplier', 20)
-----
('instance_arguments', frozenset({('n', 5)}), 'formulation_arguments//log_multiplier', 40)
-----
('instance_arguments', frozenset({('n', 5)}), 'formulation_arguments//log_multiplier', 100)
-----
('instance_arguments', frozenset({('n', 4)}), 'formulation_arguments//log_multiplier', 20)"""
    print(k)
    print("-----")

# 2) Pull out every unique log_multiplier and sort them
log_multipliers = list(sorted(set([k[3] for k in data.keys()])))

# 3) Set up x and the baseline sqrt(x)
x      = np.linspace(0, 1, 300)
y_sqrt = np.sqrt(x)

# 4) Create the figure
plt.figure(figsize=(8, 6))

# 5) Plot sqrt(x)
plt.plot(x, y_sqrt,
         linewidth=2,
         label=r'$\sqrt{x}$')

# 6) Define a set of distinctly‐styled lines (so you don’t have to fiddle with colors)
line_styles = ['--', '-.', ':', (0, (5, 1)), (0, (3, 5, 1, 5))]

# 7) Loop over your log curves
for idx, m in enumerate(log_multipliers):
    y_log = np.log(m * x + 1) / np.log(m + 1)
    ls    = line_styles[idx % len(line_styles)]
    plt.plot(x, y_log,
             linestyle=ls,
             linewidth=2,
             label=rf'$\log\left({m}x + 1\right)$'
             )

# 8) Polish
plt.title('√x vs. Normalized Logarithmic Curves', fontsize=16)
plt.xlabel('x',                    fontsize=14)
plt.ylabel('Normalized Value',      fontsize=14)
plt.grid( True, linestyle='--', alpha=0.5 )
plt.legend(title='Function', title_fontsize=12, fontsize=11, loc='lower right')
plt.tight_layout()

# 9) Show
plt.show()


# ───────────────────────────────────────────────────────────────────────────────
def plot_time_by_multiplier(data, ax):
    """
    For each log_multiplier, compute mean/std of the 'optimal' runs at each n,
    then draw an error‐bar curve plus annotate any non‐full convergences.
    """
    # collect all multipliers, sorted
    multipliers = sorted({ key[3] for key in data.keys() })
    # we’ll spread out annotation‐shifts in x so labels don’t collide
    # (in “points” units)
    x_shifts = np.linspace(-15, 15, len(multipliers))

    for m, x_shift in zip(multipliers, x_shifts):
        n_vals, means, stds = [], [], []

        for key, runs in data.items():
            if key[3] != m:
                continue

            # extract n
            n = iter(key[1]).__next__()[1]

            total = len(runs)
            good  = [r for r in runs if r.results['termination_condition']=='optimal']
            suc   = len(good)

            # mean & std, or timeout
            if suc>0:
                mu   = sum(r.results['time'] for r in good)/suc
                sigma= (sum((r.results['time']-mu)**2 for r in good)/suc)**0.5
            else:
                mu, sigma = 300, 0

            n_vals.append(n)
            means.append(mu)
            stds.append(sigma)

            # annotate only if not all succeeded
            if suc != total:
                ax.annotate(f"{suc}/{total} converged",
                            (n, mu),
                            xytext=(x_shift, 10),
                            textcoords='offset points',
                            ha='center', va='bottom',
                            fontsize=11,
                            color=f"C{multipliers.index(m)}",
                            bbox=dict(boxstyle="round,pad=0.3",
                                      fc="white", alpha=0.8,
                                      ec=f"C{multipliers.index(m)}")
                           )

        # sort by n
        xs, ys, es = zip(*sorted(zip(n_vals, means, stds)))
        ax.errorbar(xs, ys, yerr=es,
                    label=f"log_mult={m}",
                    marker='o', capsize=4,
                    linestyle='-')

# ───────────────────────────────────────────────────────────────────────────────
# (2) make the new plot
fig, ax = plt.subplots(figsize=(8,5))
plot_time_by_multiplier(data, ax)

ax.set_xlabel('n',       fontsize=14)
ax.set_ylabel('time (s)',fontsize=14)
ax.set_xticks(sorted({ next(v for k,v in key[1]) for key in data.keys() }))
ax.set_ylim(0, 310)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(title='Multiplier', fontsize=11, title_fontsize=12, loc='upper left')

fig.tight_layout()
plt.show()
fig.savefig('runtime_by_log_multiplier.png', dpi=300, bbox_inches='tight')