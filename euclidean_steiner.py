import pulp
import itertools
import cvxpy as cp




def solve(terminals, pt_c, DIM):
    candidate_steiners = list(range(len(terminals), len(terminals) * 2 - 1))
    all_points = terminals + candidate_steiners
    edges = [(i, j) for i in all_points for j in all_points if i < j]

    print(f"#terminals: {len(terminals)}")
    print(f"#candidate steiners: {len(candidate_steiners)}")
    print(f"#all points: {len(all_points)}")
    print(f"#edges: {len(edges)}")
    # edges cost go into the objective

    # cvxpy variables for each edge and each point

    x = {edge: cp.Variable(boolean=True) for edge in edges}
    y = {point: cp.Variable(boolean=True) for point in all_points}

    # Define point coordinates
    pt_coord = {t: cp.Constant(pt_c[t]) for t in terminals}
    pt_coord.update({s: cp.Variable(DIM) for s in candidate_steiners})

    # Constraints
    constraints = [y[t] == 1 for t in terminals]

    # add the constraints to the LP problem
    for point in all_points:
        constraints.append(cp.sum([x[edge] for edge in edges if point in edge]) >= y[point])
        constraints.append(cp.sum([x[edge] for edge in edges if point in edge]) <= 1000 * y[point])

    constraints.append(cp.sum([x[edge] for edge in edges]) == cp.sum([y[point] for point in all_points]) - 1)

    # for each subset S of points, add the constraint that the number of edges
    # leaving S is at least |S| - 1
    for subset in itertools.chain.from_iterable(
            itertools.combinations(all_points, r) for r in range(len(all_points) + 1)):
        if len(subset) > 1 and len(subset) < len(all_points):
            z = cp.Variable(1)
            constraints.append(cp.sum([x[edge] for edge in edges if all(point in subset for point in edge)]) <= z)
            constraints.append(z >= cp.sum([y[point] for point in subset]) - 1)

    # add the objective function to the LP problem

    norm_vars = {edge: cp.norm(pt_coord[edge[0]] - pt_coord[edge[1]]) for edge in edges}
    norms_enf = {edge: cp.Variable() for edge in edges}

    M = 1e3
    for edge in edges:
        constraints.append(norms_enf[edge] >= norm_vars[edge] - M * (1 - x[edge]))
        constraints.append(norms_enf[edge] <= M * x[edge])
        constraints.append(norms_enf[edge] <= norm_vars[edge])

    objective = cp.Minimize(cp.sum(list(norms_enf.values())))

    prob = cp.Problem(objective, constraints)

    # solve the LP problem
    prob.solve(verbose=True)

    # print the solution
    print("Objective value:", prob.value)


DIM = 2
terminals = [0, 1, 2]
pt_c = {0: [0, 0], 1: [1, 0], 2: [0, 1]}

solve(terminals, pt_c, DIM)