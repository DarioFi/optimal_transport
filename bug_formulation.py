from re import purge

import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from formulations.utils import get_lower_bound


def norm(p1, p2):
    return sum((x - y) ** 2 for x, y in zip(p1, p2)) ** 0.5


def norm_y(p1, p2, edge):
    # assert abs(edge) < 1e-8 or abs(edge - 1) < 1e-8
    return sum((x * edge - y * edge) ** 2 for x, y in zip(p1, p2)) ** 0.5


def bug_attempt(terminals):
    model = pyo.ConcreteModel()

    P = len(terminals)
    S = len(terminals) - 2
    D = len(terminals[0])  # Dimension of points

    model.P = pyo.RangeSet(0, P - 1)
    model.S = pyo.RangeSet(P, P + S - 1)
    model.D = pyo.RangeSet(0, D - 1)

    model.E1 = [(i, j) for i in model.P for j in model.S]
    model.E = model.E1

    # model.x = pyo.Var(model.S, model.D, domain=pyo.Reals)
    model.x = pyo.Var(model.S, model.D, domain=pyo.Reals)

    model.norm = pyo.Var(model.E, domain=pyo.NonNegativeReals)

    model.inside_norm = pyo.Var(model.E, model.D, domain=pyo.NonNegativeReals)

    def inside_norm_sq_constraint(model, i, j, d):
        return model.inside_norm[i, j, d] == (terminals[i][d] - model.x[j, d])

    model.inside_norm_sq_constraint = pyo.Constraint(model.E, model.D, rule=inside_norm_sq_constraint)

    def norm_e2_constraint(model, i, j):
        return model.norm[i, j] >= sum(model.inside_norm[i, j, d] ** 2 for d in model.D) ** 0.5

    model.norm_constraint = pyo.Constraint(model.E, rule=norm_e2_constraint)

    def objective_rule(model):
        return sum(model.norm[i, j] for (i, j) in model.E)

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    return model


if __name__ == '__main__':
    terminals = [(0.1, 0.9), (0.9, 0.1), (0.9, 0.9)]

    model = bug_attempt(terminals)
    # model.pprint()

    opt = SolverFactory('baron')
    results = opt.solve(model, tee=True, keepfiles=True)

    for i in model.S:
        for d in model.D:
            print(f"x[{i},{d}] = {model.x[i, d].value}")
