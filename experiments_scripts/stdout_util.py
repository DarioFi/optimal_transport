import os

example_out = """===========================================================================
 BARON version 24.5.8. Built: LNX-64 Wed May 8 10:06:40 EDT 2024 
 Running on machine RedmiDario

 BARON is a product of The Optimization Firm.
 For information on BARON, see https://minlp.com/about-baron
 Licensee: Aida Khajavirad at TOF, aida.khajavirad@gmail.com.

 If you publish work using this software, please cite publications from
 https://minlp.com/baron-publications, such as: 

 Khajavirad, A. and N. V. Sahinidis,
 A hybrid LP/NLP paradigm for global optimization relaxations,
 Mathematical Programming Computation, 10, 383-421, 2018.
===========================================================================
 This BARON run may utilize the following subsolver(s)
 For LP/MIP/QP: CLP/CBC                                         
 For NLP: IPOPT, FILTERSQP
===========================================================================
 Doing local search
 Solving bounding LP
 Preprocessing found feasible solution with value 4.10237
 Starting multi-start local search
 Done with local search
===========================================================================
  Iteration    Open nodes         Time (s)    Lower bound      Upper bound
          1             1             0.46      1.02470          4.10237                 
*         5             3             0.65      1.02470          1.69576                 
*         5             3             0.72      1.02470          1.69566                 
*        18             4             1.30      1.35047          1.59359                 
*        18             4             1.31      1.35047          1.58228                 
*        18             4             1.38      1.35047          1.58227                 
         29             0             1.45      1.58227          1.58227                 

                         *** Normal completion ***            

 Wall clock time:                     1.46
 Total CPU time used:                 1.45

 Total no. of BaR iterations:      29
 Best solution found at node:      18
 Max. no. of nodes in memory:       4
 
 All done
===========================================================================
 """
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

import numpy as np
import matplotlib.pyplot as plt


def parse(result_string):
    times = []
    lower_bounds = []
    upper_bounds = []
    open_nodes = []
    result_string = result_string.replace("*", "").replace("+", "")

    for line in result_string.strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:

            try:
                iter_num = int(parts[0])
                open_node = int(parts[1])
                time = float(parts[2])
                lower = float(parts[3])
                upper = float(parts[4])

                times.append(time)
                lower_bounds.append(lower)
                upper_bounds.append(upper)
                open_nodes.append(open_node)
            except ValueError:
                continue

    return times, lower_bounds, upper_bounds, open_nodes


def create_graph(results, title):
    times, lower_bounds, upper_bounds, open_nodes = parse(results)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left axis: lower and upper bounds
    color_lb = 'tab:green'
    color_ub = 'tab:blue'

    ax1.step(times, lower_bounds, label='Lower Bound', color=color_lb, linewidth=2, marker="^", linestyle="-",
             where="post")

    # if upper_bounds > 100 then i want to show like an infinity

    upper_bounds_plot = [ub if ub <= 100 else np.inf for ub in upper_bounds]
    ax1.step(times, upper_bounds_plot, label='Upper Bound', color=color_ub, linewidth=2, marker="^", linestyle="-",
             where="post")

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Bound Value')
    ax1.tick_params(axis='y')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color_nodes = 'tab:red'



    # Prepare and sort data
    x = np.array(times)
    y = np.array(open_nodes)
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]

    # Interpolate linearly, then smooth with Savitzky-Golay filter
    interp_fn = interp1d(x_sorted, y_sorted, kind='linear', fill_value="extrapolate")
    x_fine = np.linspace(x_sorted.min(), x_sorted.max(), 500)
    y_interp = interp_fn(x_fine)

    # Apply Savitzky-Golay filter (window must be odd and < len(y_interp))
    y_smooth = savgol_filter(y_interp, window_length=21, polyorder=3)

    # Plot the smoothed curve
    ax2.plot(x_fine, y_smooth, label='Open Nodes (Interpolated)', color=color_nodes, linewidth=2, linestyle="--")

    # Optional: also plot raw data
    ax2.plot(x_sorted, y_sorted, color=color_nodes, marker='^', linestyle="", label='Open Nodes (raw)')

    ax2.set_ylabel('Open Nodes')
    ax2.tick_params(axis='y')

    # Right axis: open nodes
    # ax2.plot(times, open_nodes, label='Open Nodes', color=color_nodes, marker='^', linewidth=2, linestyle="")
    # ax2.set_ylabel('Open Nodes')
    # ax2.tick_params(axis='y')

    # Add second legend manually
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    ax1.set_ylim(0, max([u for u in upper_bounds if u < 100]) + 1)

    # Title and grid
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(title + '.png', dpi=300)
    plt.show()

def create_subplot(ax, results, title):
    """
    Draws on `ax` and its twin, but does NOT call legend().
    Returns (ax, ax2) so we can grab handles later.
    """
    times, lower_bounds, upper_bounds, open_nodes = parse(results)

    # primary axis
    ax.step(times, lower_bounds, label='Lower Bound',
            color='tab:green', linewidth=1, marker="^", linestyle="-", where="post")
    ub_plot = [u if u <= 100 else np.inf for u in upper_bounds]
    ax.step(times, ub_plot, label='Upper Bound',
            color='tab:blue', linewidth=1, marker="^", linestyle="-", where="post")
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Bound Value')
    ax.grid(True)

    # secondary axis
    ax2 = ax.twinx()
    x = np.array(times); y = np.array(open_nodes)
    idx = np.argsort(x); x_s, y_s = x[idx], y[idx]
    fn = interp1d(x_s, y_s, kind='linear', fill_value="extrapolate")
    x_fine = np.linspace(x_s.min(), x_s.max(), 200)
    y_i = fn(x_fine)
    y_sm = savgol_filter(y_i, window_length=21, polyorder=3)

    ax2.plot(x_fine, y_sm, label='Open Nodes (smoothed)',
             linestyle="--", linewidth=1, color='tab:orange')
    ax2.plot(x_s, y_s, label='Open Nodes (raw)',
             linestyle="", marker="o", color='tab:orange')
    ax2.set_ylabel('Open Nodes')

    return ax, ax2


if __name__ == "__main__":
    folder = "starting_point_strings/"
    patterns = [
        ("nostartingpoint", "no_starting_point_experiment_iter={}.txt"),
        ("startingpoint",    "starting_point_experiment_iter={}.txt"),
    ]

    nrows = 3
    ncols = 3
    # Precompute shared axis limits per iter (0–2)
    axis_ranges = {}
    for i in range(nrows*ncols):
        all_t, all_b, all_n = [], [], []
        for _, pat in patterns:
            fn = os.path.join(folder, pat.format(i))
            with open(fn) as f:
                t, lb, ub, nodes = parse(f.read())
            all_t += t
            all_b += lb + [u for u in ub if u <= 100]
            all_n += nodes
        axis_ranges[i] = {
            'xlim':  (0, max(all_t)),
            'ylim':  (0, max(all_b) + 1),
            'y2lim': (min(all_n), max(all_n)),
        }

    for group_name, pattern in patterns:
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12))
        subplot_axes = []  # to collect (ax, ax2)
        axes_flat = axes.flatten()  # <-- flatten here

        for i, ax in enumerate(axes_flat):
            fn = os.path.join(folder, pattern.format(i))
            with open(fn) as f:
                content = f.read()

            ax, ax2 = create_subplot(ax, content, title=f"{group_name} iter={i}")
            # sync limits
            xr = axis_ranges[i]['xlim']
            yr = axis_ranges[i]['ylim']
            y2r = axis_ranges[i]['y2lim']
            ax.set_xlim(*xr)
            ax.set_ylim(*yr)
            ax2.set_ylim(*y2r)

            subplot_axes.append((ax, ax2))

            if i == 0:
                # plot legend
                # Add second legend manually
                ax1_lines, ax1_labels = ax.get_legend_handles_labels()
                ax2_lines, ax2_labels = ax2.get_legend_handles_labels()
                ax2.legend(ax1_lines + ax2_lines,
                           ax1_labels + ax2_labels,
                           loc='upper left')

        # now pull handles & labels once from the first subplot pair
        h1, l1 = subplot_axes[0][0].get_legend_handles_labels()
        h2, l2 = subplot_axes[0][1].get_legend_handles_labels()
        handles, labels = h1 + h2, l1 + l2

        # place a single legend for the whole figure
        fig.legend(handles, labels,
                   loc='upper center',
                   ncol=len(labels),
                   frameon=False,
                   bbox_to_anchor=(0.5, 1.05))

        # set title
        if group_name == "nostartingpoint":
            fig.suptitle(f"No initialization")
        else:
            fig.suptitle(f"Optimal starting point")

        plt.tight_layout()
        out_png = f"{group_name}_3x3.png"
        fig.savefig(out_png, dpi=600, bbox_inches="tight")
        print(f"Saved {out_png}")
        plt.show()
        plt.close(fig)