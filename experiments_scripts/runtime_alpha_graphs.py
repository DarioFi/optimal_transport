import matplotlib.pyplot as plt
from opt_trans.data_handling.experiment_data import Database

folder = '../run_logs/runtime_alpha/'
db = Database.populate_from_folder(folder)
data = db.index_on('instance_arguments')

print(data.keys())
# data key is a Frozenset{ (("alpha", alpha), ("n", n)) }
# I want to extract the average runtime for each alpha value
# and plot it

alphas = []
means = []
stds = []
success_rates = []

for key, runs in data.items():
    key = list(key[1])
    for (name, value) in key:
        if name == 'alpha':
            alpha = value
        elif name == 'n':
            n = value
    good = [r for r in runs if r.results['termination_condition'] == 'optimal']
    tot = len(runs)
    succ = len(good)

    # compute mean/std or use timeout
    if succ > 0:
        mean_t = sum(r.results['time'] for r in good) / succ
        std_t = (sum((r.results['time'] - mean_t) ** 2 for r in good) / succ) ** 0.5
        success_rate = succ / tot
    else:
        mean_t, std_t = 300, 0
        success_rate = 0

    print(f"alpha: {alpha}, n: {n}, mean: {mean_t}, std: {std_t}")


    alphas.append(alpha)
    means.append(mean_t)
    stds.append(std_t)
    success_rates.append(success_rate)

# sort
sorted_indices = sorted(range(len(alphas)), key=lambda k: alphas[k])
alphas = [alphas[i] for i in sorted_indices]
means = [means[i] for i in sorted_indices]
stds = [stds[i] for i in sorted_indices]
success_rates = [success_rates[i] for i in sorted_indices]

# plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(alphas, means, yerr=stds, fmt='^', capsize=5, label='Mean runtime')

# also plot success rate

ax2 = ax.twinx()
ax2.plot(alphas, success_rates, '^-', label='Success rate', color='red')
ax2.set_ylabel('Success rate')
ax2.set_ylim(0, 1.1)
ax2.legend(loc='upper right')


ax.set_xlabel(r'$\alpha$')
ax.set_ylabel('Runtime (s)')
ax.set_title(r'Runtime vs $\alpha$')
ax.set_xticks(alphas)
ax.set_xticklabels([f"{a:.2f}" for a in alphas])
ax.legend()
ax.grid()
plt.tight_layout()
plt.savefig('runtime_alpha.png')
plt.show()
