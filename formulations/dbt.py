import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from formulations.utils import get_lower_bound


def norm(p1, p2):
    return sum((x - y) ** 2 for x, y in zip(p1, p2)) ** 0.5


def dbt(terminals, masses, maximum_degree, alpha, use_bind_first_steiner, use_obj_lb, use_convex_hull):
    assert len(terminals) == len(masses)
    assert abs(sum(masses)) < 1e-7
    assert masses[0] < 0
    assert all(m > 0 for m in masses[1:])

    model = pyo.ConcreteModel()

    P = len(terminals)
    S = len(terminals) - 2
    D = len(terminals[0])  # Dimension of points

    model.P = pyo.RangeSet(0, P - 1)
    model.S = pyo.RangeSet(P, P + S - 1)
    model.D = pyo.RangeSet(0, D - 1)

    model.E1 = [(i, j) for i in model.P for j in model.S]
    model.E2 = [(i, j) for i in model.S for j in model.S if i < j]
    model.E = model.E1 + model.E2

    model.x = pyo.Var(model.S, model.D, domain=pyo.Reals)
    model.y = pyo.Var(model.E, domain=pyo.Binary)
    model.f = pyo.Var(model.E, domain=pyo.NonNegativeReals)

    # terminals have degree 1 constraint
    def degree_one_constraint(model, i):
        return sum(model.y[i, j] for j in model.S) == 1

    model.degree_one_constraint = pyo.Constraint(model.P, rule=degree_one_constraint)

    # connectivity constraint
    def connectivity_constraint(model, i):
        if i != min(model.S):
            return sum(model.y[j, i] for j in model.S if j < i) == 1
        return pyo.Constraint.Skip

    model.connectivity_constraint = pyo.Constraint(model.S, rule=connectivity_constraint)

    def flow_conservation_terminal_constraint(model, i):
        if i == 0:
            # note: the first terminal is special so the flow is in the opposite direction
            # we still are not sure how to handle this cleanly but this should work
            return sum(model.f[i, j] for j in model.S) == -masses[i]
        else:
            return sum(model.f[i, j] for j in model.S) == masses[i]

    model.flow_conservation_terminal_constraint = pyo.Constraint(model.P, rule=flow_conservation_terminal_constraint)

    def flow_conservation_steiner_constraint(model, i):
        return sum(model.f[j, i] for j in model.S if j < i) - sum(model.f[i, j] for j in model.S if j > i) - sum(
            model.f[j, i] for j in model.P if j != 0) + model.f[0, i] == 0

    model.flow_conservation_steiner_constraint = pyo.Constraint(model.S, rule=flow_conservation_steiner_constraint)

    def degree_constraint(model, i):
        return sum(model.y[k, j] for (k, j) in model.E if i in (k, j)) == maximum_degree

    model.degree_constraint = pyo.Constraint(model.S, rule=degree_constraint)

    def flow_activation_constraint(model, i, j):
        return model.f[i, j] <= model.y[i, j]

    model.flow_activation_constraint = pyo.Constraint(model.E, rule=flow_activation_constraint)

    if use_obj_lb:
        raise NotImplementedError("Objective lower bound not implemented yet")

    if use_bind_first_steiner:

        # bind the source to the first steiner point

        model.y[0, min(model.S)].fix(1)
        model.f[0, min(model.S)].fix(-masses[0])

    if use_convex_hull:
        model.c = pyo.Var(model.S, model.P, domain=pyo.NonNegativeReals)

        # constraint is that x_i = sum_{j in P} c_{ij} p_j
        # sum_{j in P} c_{ij} = 1

        def convex_hull_constraint(model, i, d):
            return model.x[i, d] == sum(model.c[i, j] * terminals[j][d] for j in model.P)

        model.convex_hull_constraint = pyo.Constraint(model.S, model.D, rule=convex_hull_constraint)

        def convex_hull_sum_constraint(model, i):
            return sum(model.c[i, j] for j in model.P) == 1

        model.convex_hull_sum_constraint = pyo.Constraint(model.S, rule=convex_hull_sum_constraint)

    def objective_rule(model):
        return sum(
            model.f[i, j] ** alpha * norm(terminals[i], [model.x[j, d] for d in model.D]) for (i, j) in model.E1) + sum(
            model.f[i, j] ** alpha * norm([model.x[i, d] for d in model.D], [model.x[j, d] for d in model.D]) for (i, j)
            in model.E2)

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    return model


if __name__ == '__main__':
    terminals = [(0, 1), (1, 0), (1, 1)]
    masses = [-1, .5, .5]
    maximum_degree = 3
    alpha = .5

    bind_first_steiner = False
    use_obj_lb = False
    use_convex_hull = False

    model = dbt(terminals, masses, maximum_degree, alpha, bind_first_steiner, use_obj_lb, use_convex_hull)

    model.pprint()

    solver = SolverFactory('baron')
    results = solver.solve(model, tee=True)
    print(results)
