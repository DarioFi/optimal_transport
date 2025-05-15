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

    from scipy.interpolate import interp1d
    from scipy.signal import savgol_filter

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
    plt.show()


if __name__ == "__main__":
    n = 7
    with open(f"result_string_{n=}.txt") as f:
        res = "".join(f.readlines())

    print(res)


    create_graph(res, f"Solution trajectory {n=}")
