import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from opt_trans.formulations.dbt import dbt_alpha_0
from opt_trans.problems.instance_generators import *

eps = 1e-1

points_P_dict = get_5_points_computations(eps)

points_P = points_P_dict["terminals"]

l1 = len(points_P_dict["terminals"]) - 1
points_P_dict["alpha"] = 0
points_P_dict["masses"] = [-1] + [1 / l1] * l1

model = dbt_alpha_0(**points_P_dict, use_bind_first_steiner=True, use_obj_lb=False, use_gurobi=False,
                    use_better_obj=True,
                    use_convex_hull=True)

solver = SolverFactory('baron')

results = solver.solve(model, tee=True, options_string="maxtime=7200")

if results.solver.status == pyo.SolverStatus.ok:
    print(f"Optimal objective value: {model.obj()}")

    steiner_solution = [[pyo.value(model.x[j, d]) for d in model.D] for j in model.S]

else:
    print("Solver status:", results.solver.status)
    exit()

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# ——— Start of new plotting snippet ———
plt.style.use('seaborn-v0_8-white')
fig, ax = plt.subplots(figsize=(8, 6))

# (Optional) draw convex hull around terminals
pts = np.array(points_P)
hull = ConvexHull(pts)
for simplex in hull.simplices:
    ax.plot(pts[simplex, 0], pts[simplex, 1],
            linestyle='--', color='gray', linewidth=1)

# draw edges first (so points are on top)
for i in model.P:
    for j in model.S:
        if pyo.value(model.y[i, j]) > 0.5:
            ste_idx = j - len(points_P)
            xvals = [points_P[i][0], steiner_solution[ste_idx][0]]
            yvals = [points_P[i][1], steiner_solution[ste_idx][1]]
            ax.plot(xvals, yvals, linewidth=2, color='dimgray', zorder=1)
for i in model.S:
    for j in model.S:
        if j > i and pyo.value(model.y[i, j]) > 0.5:
            i0, j0 = i - len(points_P), j - len(points_P)
            xvals = [steiner_solution[i0][0], steiner_solution[j0][0]]
            yvals = [steiner_solution[i0][1], steiner_solution[j0][1]]
            ax.plot(xvals, yvals, linewidth=2, color='dimgray', zorder=1)

# scatter terminals and Steiner points
ax.scatter(pts[:, 0], pts[:, 1],
           s=100, c='C0', marker='o', edgecolor='k', label='Terminals', zorder=3)
sp = np.array(steiner_solution)
ax.scatter(sp[:, 0], sp[:, 1],
           s=120, c='C3', marker='^', edgecolor='k', label='Steiner points', zorder=4)

# annotate with a slight offset so labels don't overlap markers
for i, (x, y) in enumerate(points_P):
    ax.annotate(f"T{i}", (x, y),
                textcoords="offset points", xytext=(3, 3),
                fontsize=10, color='C0')
for i, (x, y) in enumerate(steiner_solution):
    ax.annotate(f"S{i}", (x, y),
                textcoords="offset points", xytext=(3, -8),
                fontsize=10, color='C3')

ax.set_aspect('equal', 'box')
# ax.set_title("Optimal Steiner Tree", fontsize=14)
ax.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.savefig("experiments_scripts/counterexample_steiner_tree.png", dpi=300)
plt.show()


# ——— End of new plotting snippet ———


# compute distance matrix
def compute_distance_matrix(points):
    n = len(points)
    distance_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            distance = norm(points[i], points[j], range(len(points[i])))
            distance_matrix[i][j] = distance
            distance_matrix[j][i] = distance
    return distance_matrix


distance_matrix = np.array(compute_distance_matrix(points_P))
print(distance_matrix)

triangle_cost = (distance_matrix[0][2]) * math.sqrt(3)
closest_cost = distance_matrix[0, 3]
print(2 * triangle_cost + closest_cost)
