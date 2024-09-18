import math
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from random import random
from formulations.p6_ouzia_maculan_alpha import p6_model_alpha
from problems.alpha_tests import example, get_5_masses

eps = 1e-1

terminals, masses = example()

alpha = .5

model = p6_model_alpha(
    terminals,
    masses,
    alpha
)

# solver = SolverFactory('baron')
solver = SolverFactory("scip")

solver.options['limits/gap'] = 0.0001
results = solver.solve(model, tee=True, options_string="maxtime=7200")

if results.solver.status == pyo.SolverStatus.ok:
    print(f"Optimal objective value: {model.Obj()}")

    steiner_solution = [[pyo.value(model.x[j, d]) for d in model.D] for j in model.S]
    print(f"Steiner points: {steiner_solution}")

    flows = [[pyo.value(model.f[i, j]) for j in model.S.union(model.P)] for i in model.S.union(model.P)]
    abs_flows = [[pyo.value(model.abs_f[i, j]) for j in model.S.union(model.P)] for i in model.S.union(model.P)]
    print(f"Flows: {flows}")
    print(f"AbsFw: {abs_flows}")
    print(f"Flow from node 0 {sum(flows[0])} {flows[0]}")

# plot points
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

for i, p in enumerate(terminals):  # with labels
    ax.scatter(*p, color='blue')
    ax.text(*p, s=f"{i}", fontsize=12)

for i, p in enumerate(steiner_solution):
    ax.scatter(*p, color='red')
    ax.text(*p, s=f"{i}", fontsize=12)

# print edges between points
for i in model.P:
    for j in model.S:
        if pyo.value(model.y_p_s[i, j]) > 0.5:
            print(f"Edge between {i} and {j} with mass {pyo.value(model.f[i, j])}")
            j_new = j - len(terminals)

            f_abs = pyo.value(model.abs_f[i, j])
            ax.plot([terminals[i][0], steiner_solution[j_new][0]], [terminals[i][1], steiner_solution[j_new][1]],
                    color="black", alpha=pyo.value(model.abs_f[i, j]))

for i in model.S:
    for j in model.S:
        if pyo.value(model.y_s_s[i, j]) > 0.5:
            print(f"Edge between {i} and {j} with mass {pyo.value(model.f[i, j])}")

            j_new = j - len(terminals)
            i_new = i - len(terminals)


            ax.plot([steiner_solution[i_new][0], steiner_solution[j_new][0]],
                    [steiner_solution[i_new][1], steiner_solution[j_new][1]], color="black", alpha=pyo.value(model.abs_f[i, j]))

ax.set_aspect('equal', adjustable='datalim')
plt.grid()
plt.show()
