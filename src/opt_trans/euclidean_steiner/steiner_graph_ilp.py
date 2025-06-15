# implementation of an ILP solver for the steiner graph problem, a simplification of the Euclidean Steiner Tree problem


import plot_utils
import pulp
import matplotlib.pyplot as plt
import point_generator

prob = pulp.LpProblem("Steiner Tree", pulp.LpMinimize)

DIM = 2

points, pt_c, terminals, edges = point_generator.gen_all(12, 30, 6, DIM)


def distance(p1, p2):
    return sum((p1[i] - p2[i]) ** 2 for i in range(DIM)) ** 0.5


edges_cost = {edge: distance(pt_c[edge[0]], pt_c[edge[1]]) for edge in edges}
print(edges_cost)

# create a new LP variable for each edge
x = {}
for edge in edges:
    x[edge] = pulp.LpVariable("x" + str(edge), 0, 1, pulp.LpBinary)

# create a new LP variable for each point
y = {}
for point in points:
    y[point] = pulp.LpVariable("y" + str(point), 0, 1, pulp.LpBinary)

for t in terminals:
    prob += y[t] == 1

# add the objective function to the LP problem
prob += pulp.lpSum(x[edge] * edges_cost[edge] for edge in edges)

# add the constraints to the LP problem
for point in points:
    prob += pulp.lpSum(x[edge] for edge in edges if point in edge) >= y[point]
    prob += pulp.lpSum(x[edge] for edge in edges if point in edge) <= 1000 * y[point]

prob += pulp.lpSum(x[edge] for edge in edges) == pulp.lpSum(y[point] for point in points) - 1

# for each subset S of points, add the constraint that the number of edges
# leaving S is at least |S| - 1
import itertools


def powerset(iterable):
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))


for subset in powerset(points):
    if len(subset) > 1 and len(subset) < len(points):
        # new variable that is max(0, pulp.lpSum(y[point] for point in subset) - 1)
        z = pulp.LpVariable("z" + str(subset), 0, len(subset) - 1, pulp.LpContinuous)
        prob += pulp.lpSum(x[edge] for edge in edges if all(point in subset for point in edge)) <= z
        prob += z >= pulp.lpSum(y[point] for point in subset) - 1

prob.solve()
print("Objective value:", pulp.value(prob.objective))
print("Is feasible?", pulp.LpStatus[prob.status] == "Optimal")


plot_utils.plot_points_on_plane([pt_c[t] for t in terminals], terminals, 'r^')

not_terminals_sel_pt = [
    p for p in points if p not in terminals and pulp.value(y[p]) == 1
]
not_terminals_not_sel_pt = [
    p for p in points if p not in terminals and pulp.value(y[p]) == 0
]

plot_utils.plot_points_on_plane([pt_c[p] for p in not_terminals_sel_pt], not_terminals_sel_pt, color='bo', alpha=1)

plot_utils.plot_points_on_plane([pt_c[p] for p in not_terminals_not_sel_pt], not_terminals_not_sel_pt, color='bo',
                                alpha=.2)

selected_edges = [edge for edge in edges if pulp.value(x[edge]) == 1]
plot_utils.plot_edges_on_plane([[pt_c[e[0]], pt_c[e[1]]] for e in selected_edges])

not_selected_edges = [edge for edge in edges if pulp.value(x[edge]) == 0]
plot_utils.plot_edges_on_plane([[pt_c[e[0]], pt_c[e[1]]] for e in not_selected_edges],
                               alpha=0.1)

plt.show()

