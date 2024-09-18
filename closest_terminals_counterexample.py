import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from formulations.mmx import mmx_model
from formulations.p6_ouzia_maculan import create_p6_model
from problems.closest_counterexample import *

eps = 1e-1

# points_P, initial_steiner_points = get_7_points(eps)
# points_P, initial_steiner_points = get_5_points_eps_0_1()
# points_P, initial_steiner_points = hypercube()
points_P = random_points_unit_square(6)

model = mmx_model(points_P)
# model = create_p6_model(points_P, initial_steiner_points)

solver = SolverFactory('baron')
# solver = SolverFactory("scip")


results = solver.solve(model, tee=True, options_string="maxtime=7200")

if results.solver.status == pyo.SolverStatus.ok:
    print(f"Optimal objective value: {model.Obj()}")

    steiner_solution = [[pyo.value(model.x[j, d]) for d in model.D] for j in model.S]
    print(f"Steiner points: {steiner_solution}")

# plot points
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

for i, p in enumerate(steiner_solution):
    ax.scatter(*p, color='red')
    ax.text(*p, s=f"{i}", fontsize=12)

for i, p in enumerate(points_P):  # with labels
    ax.scatter(*p, color='blue')
    ax.text(*p, s=f"{i}", fontsize=12)

# print edges between points
for i in model.P:
    for j in model.S:
        if pyo.value(model.y_p_s[i, j]) > 0.5:
            print(f"Edge between {i} and {j}")
            j = j - len(points_P)
            ax.plot([points_P[i][0], steiner_solution[j][0]], [points_P[i][1], steiner_solution[j][1]],
                    color='black')

for i in model.S:
    for j in model.S:
        if pyo.value(model.y_s_s[i, j]) > 0.5:
            print(f"Edge between {i} and {j}")

            j = j - len(points_P)
            i_new = i - len(points_P)

            ax.plot([steiner_solution[i_new][0], steiner_solution[j][0]],
                    [steiner_solution[i_new][1], steiner_solution[j][1]], color='black')

ax.set_aspect('equal', adjustable='datalim')
plt.grid()
plt.show()

dist_0_2 = norm(points_P[0], points_P[2], range(len(points_P[0])))
print(dist_0_2)

print(2 * (dist_0_2) * math.sqrt(3))
