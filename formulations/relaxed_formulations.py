import itertools

import pyomo.environ as pyo


def norm(p1, p2):
    return sum((x - y) ** 2 for x, y in zip(p1, p2)) ** 0.5


class pyomo_wrapper:
    """
    Wrapper class to stop pyomo from flattening tuples
    """

    def __init__(self, data):
        self.data = data

    def __str__(self):
        return str(self.data)


def add_bilinear(model, z, x, y, index_sets, relaxed: bool, constraint_name: str):
    """
    Add constraint z[i_z] = x[i_x] * y[i_y] for (i_z, i_x, i_y) in index_sets
    :param index_sets:
    :param model:
    :param z:
    :param x:
    :param y:
    :param relaxed:
    :param constraint_name:
    :return:
    """

    index_wrapped = [pyomo_wrapper(i) for i in index_sets]

    if relaxed is False:
        def bilinear_constraint_rule(model, ind):
            i_z, i_x, i_y = ind.data
            return z[i_z] == x[i_x] * y[i_y]

        model.add_component(constraint_name, pyo.Constraint(index_wrapped, rule=bilinear_constraint_rule))

    else:
        # instead of having z = x * y we now need
        # z >= 0
        # z >= x + y - 1
        # z <= x
        # z <= y

        def bilinear_constraint_rule_1(model, ind):
            i_z, i_x, i_y = ind.data
            return z[i_z] >= 0

        def bilinear_constraint_rule_2(model, ind):
            i_z, i_x, i_y = ind.data
            return z[i_z] >= x[i_x] + y[i_y] - 1

        def bilinear_constraint_rule_3(model, ind):
            i_z, i_x, i_y = ind.data
            return z[i_z] <= x[i_x]

        def bilinear_constraint_rule_4(model, ind):
            i_z, i_x, i_y = ind.data
            return z[i_z] <= y[i_y]

        model.add_component(constraint_name + "_1", pyo.Constraint(index_wrapped, rule=bilinear_constraint_rule_1))
        model.add_component(constraint_name + "_2", pyo.Constraint(index_wrapped, rule=bilinear_constraint_rule_2))
        model.add_component(constraint_name + "_3", pyo.Constraint(index_wrapped, rule=bilinear_constraint_rule_3))
        model.add_component(constraint_name + "_4", pyo.Constraint(index_wrapped, rule=bilinear_constraint_rule_4))


def dbt_relaxed_alpha0(terminals, alpha, masses, relax_y: bool, relax_w: bool, disjunctive_w: bool):
    assert alpha == 0
    P = len(terminals)
    S = len(terminals) - 2
    D = len(terminals[0])  # Dimension of points

    model = pyo.ConcreteModel()

    model.P = pyo.RangeSet(0, P - 1)
    model.S = pyo.RangeSet(P, P + S - 1)
    model.D = pyo.RangeSet(0, D - 1)

    model.E1 = [(i, j) for i in model.P for j in model.S]
    model.E2 = [(i, j) for i in model.S for j in model.S if i < j]
    model.E = model.E1 + model.E2

    terminals_dict = {(i, d): terminals[i][d] for i in range(P) for d in range(D)}
    model.x = pyo.Var(model.S, model.D, domain=pyo.Reals, bounds=(0, 1))

    if relax_y:
        model.y = pyo.Var(model.E, domain=pyo.NonNegativeReals, bounds=(0, 1))
    else:
        model.y = pyo.Var(model.E, domain=pyo.Binary)

    # region Y-polytope

    def terminals_leaf_constraint(model, i):
        return sum(model.y[i, j] for j in model.S) == 1

    model.terminals_leaf_constraint = pyo.Constraint(model.P, rule=terminals_leaf_constraint)

    def connectivity_constraint(model, j):
        if j == min(model.S): return pyo.Constraint.Skip
        return sum(model.y[i, j] for i in model.S if i < j) == 1

    model.connectivity_constraint = pyo.Constraint(model.S, rule=connectivity_constraint)

    def degree_constraint(model, i):
        return sum(model.y[j, i] for j in model.P if j != 0) + sum(
            model.y[i, j] for j in model.S if j > i
        ) == 2

    model.degree_constraint = pyo.Constraint(model.S, rule=degree_constraint)

    # bind first steiner
    model.y[0, min(model.S)].fix(1)

    # endregion

    # region X-polytope

    model.cc = pyo.Var(model.S, model.P, domain=pyo.NonNegativeReals, bounds=(0, 1))

    def convex_hull_constraint(model, i, d):
        return sum(terminals[j][d] * model.cc[i, j] for j in model.P) == model.x[i, d]

    model.convex_hull_constraint = pyo.Constraint(model.S, model.D, rule=convex_hull_constraint)

    # endregion

    # region bilinear-terms

    w_index = [(i, j) for i in model.S for j in model.S if i != j] + [(i, j) for i in model.S for j in model.P] + [
        (j, i) for i in
        model.S for j
        in model.P]

    model.w = pyo.Var(w_index, model.D, domain=pyo.NonNegativeReals, bounds=(0, 1))

    w_index_s_s = [
        ((((i, j), d), (i, d), (min(i, j), max(i, j)))) for i in model.S for j in model.S for d in model.D if
        i != j
    ]
    add_bilinear(model, model.w, model.x, model.y, w_index_s_s, relax_w, "w_bilinear_constraint_s_s")

    w_index_p_s = [
        ((((i, j), d), (i, d), (min(i, j), max(i, j)))) for i in model.P for j in model.S for d in model.D
    ]

    # note: this is never relaxed because terminals are constant and therefore this constraint is actually linear
    add_bilinear(model, model.w, terminals_dict, model.y, w_index_p_s, False, "w_bilinear_constraint_p_s")

    w_index_s_p = [
        ((((i, j), d), (i, d), (min(i, j), max(i, j)))) for i in model.S for j in model.P for d in model.D
    ]
    add_bilinear(model, model.w, model.x, model.y, w_index_s_p, relax_w, "w_bilinear_constraint_s_p")

    # endregion

    # objective function:

    def objective_rule(model):
        return sum(
            norm([model.w[(i, j), d] for d in model.D], [model.w[(j, i), d] for d in model.D]) for i in model.S for j in
            model.S if i < j
        ) + sum(
            norm([model.w[(i, j), d] for d in model.D], [model.w[(j, i), d] for d in model.D]) for i in model.P for j in
            model.S
        )

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    return model


if __name__ == '__main__':
    m = dbt_relaxed_alpha0(
        [[0, 1], [0, 0], [1, 0], [1, 1]],
        masses=[0, 0, 0, 0],
        alpha=0,
        relax_w=True,
        relax_y=False,
        disjunctive_w=False,
    )

    m.pprint()

    m = dbt_relaxed_alpha0(
        [[0, 1], [0, 0], [1, 0], [1, 1]],
        masses=[0, 0, 0, 0],
        alpha=0,
        relax_w=True,
        relax_y=False,
        disjunctive_w=False,
    )

    m.pprint()
