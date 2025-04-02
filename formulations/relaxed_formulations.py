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


def enumerate_assignments(A, B):
    """
    Enumerate all possible functions A -> B
    :return:
    """
    fs = [{
        a: b for a in A for b in b_assignemnt
    } for b_assignemnt in itertools.product(B, repeat=len(A))]
    return [pyomo_wrapper(f) for f in fs]
    # return [pyomo_wrapper(x) for x in itertools.product(range(P), repeat=S)]


def dbt_relaxed_alpha0(terminals, alpha, masses, relax_y: bool, relax_w: bool, disjunctive_w: int | bool,
                       use_geometric_cut_50: bool):
    """

    :param terminals:
    :param alpha:
    :param masses:
    :param relax_y:
    :param relax_w:
    :param disjunctive_w: how many x variables to put in the disjunctive assignemnt
    :param use_geometric_cut_50:
    :return:
    """
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

    if disjunctive_w:
        # first disjunctive_w elements of model.s

        disjunctive_s = [x for i, x in enumerate(model.S) if i < disjunctive_w]
        classic_s = [x for i, x in enumerate(model.S) if i >= disjunctive_w]

        if len(classic_s) > 0:
            classic_s = pyo.RangeSet(min(classic_s), max(classic_s))

        disjunctive_s = pyo.RangeSet(min(disjunctive_s), max(disjunctive_s))

        model.A = enumerate_assignments(disjunctive_s, model.P)
    else:
        disjunctive_s = []
        classic_s = model.S

    model.cc = pyo.Var(classic_s, model.P, domain=pyo.NonNegativeReals, bounds=(0, 1))

    def convex_hull_constraint(model, i, d):
        return sum(terminals[j][d] * model.cc[i, j] for j in model.P) == model.x[i, d]

    model.convex_hull_constraint = pyo.Constraint(classic_s, model.D, rule=convex_hull_constraint)

    if disjunctive_w:
        model.x_a = pyo.Var(model.A, disjunctive_s, model.D, domain=pyo.NonNegativeReals, bounds=(0, 1))

        model.l_a = pyo.Var(model.A, domain=pyo.NonNegativeReals, bounds=(0, 1))

        def disj_sum_x_constraint(model, d, i):
            return sum(model.x_a[a, i, d] for a in model.A) == model.x[i, d]

        model.disj_sum_constraint = pyo.Constraint(model.D, disjunctive_s, rule=disj_sum_x_constraint)

        def disj_assignment_constraint(model, a, i, d):
            return model.x_a[a, i, d] == terminals[a.data[i]][d] * model.l_a[a]

        model.disj_assignment_constraint = pyo.Constraint(model.A, disjunctive_s, model.D,
                                                          rule=disj_assignment_constraint)

        def sum_l_constraint(model):
            return sum(model.l_a[a] for a in model.A) == 1

        model.sum_l_constraint = pyo.Constraint(rule=sum_l_constraint)

        # todo finish the b_a constraints

    # endregion

    # region bilinear-terms

    # some w are unaffected by the disjunctive relaxation, those who do not depend on disjunctive_s
    # in particular those which depend on terminals are always unaffected
    # the rest are affected and need to be split into two parts

    #

    w_index = [(i, j) for i in model.S for j in model.S if i != j] + [(i, j) for i in model.S for j in model.P] + [
        (j, i) for i in
        model.S for j
        in model.P]

    # this stays the same even when using the disjunctive version
    model.w = pyo.Var(w_index, model.D, domain=pyo.NonNegativeReals, bounds=(0, 1))

    # if (i,d) is in the disjunctive_s then we need to apply it to w_a
    w_index_s_s_classic = [
        ((((i, j), d), (i, d), (min(i, j), max(i, j)))) for i in model.S for j in model.S for d in model.D if
        i != j and i in classic_s
    ]

    add_bilinear(model, model.w, model.x, model.y, w_index_s_s_classic, relax_w, "w_bilinear_constraint_s_s")

    w_index_p_s = [
        ((((i, j), d), (i, d), (min(i, j), max(i, j)))) for i in model.P for j in model.S for d in model.D
    ]

    # note: this is never relaxed because terminals are constant and therefore this constraint is actually linear
    add_bilinear(model, model.w, terminals_dict, model.y, w_index_p_s, False, "w_bilinear_constraint_p_s")

    # if (i,d) is in the disjunctive_s then we need to apply it to w_a
    w_index_s_p_classic = [
        ((((i, j), d), (i, d), (min(i, j), max(i, j)))) for i in model.S for j in model.P for d in model.D
        if i in classic_s
    ]
    # todo: we should assert that min(i,j) = j and max(i,j) = i

    add_bilinear(model, model.w, model.x, model.y, w_index_s_p_classic, relax_w, "w_bilinear_constraint_s_p")

    if disjunctive_w:
        w_index_disjunctive = [
            ((i, j), d) for i in disjunctive_s for j in model.S.union(model.P) for d in model.D
            if i != j
        ]

        w_index_disjunctive_wrap = [pyomo_wrapper(x) for x in w_index_disjunctive]

        w_index_disjunctive_2 = [(x, d) for x in w_index for d in model.D if x[0] in disjunctive_s]

        assert sorted(list(w_index_disjunctive)) == sorted(list(w_index_disjunctive_2))

        print("Assertion passed")

        model.w_a = pyo.Var(model.A, w_index_disjunctive, domain=pyo.NonNegativeReals, bounds=(0, 1))

        def disj_sum_w_constraint(model, w_ind):
            return sum(model.w_a[a, w_ind.data] for a in model.A) == model.w[w_ind.data]

        model.disj_sum_w_constraint = pyo.Constraint(w_index_disjunctive_wrap, rule=disj_sum_w_constraint)

        def w_bilinear_disj_constraint(model, a, w_ind):
            i, j = w_ind.data[0]
            d = w_ind.data[1]
            return model.w_a[a, (i, j), d] == model.l_a[a] * model.y[min(i, j), max(i, j)] * terminals_dict[
                (a.data[i], d)]

        model.w_bilinear_disj_constraint = pyo.Constraint(model.A, w_index_disjunctive_wrap,
                                                          rule=w_bilinear_disj_constraint)

    # endregion

    def objective_rule(model):
        return sum(
            norm([model.w[(i, j), d] for d in model.D], [model.w[(j, i), d] for d in model.D]) for i in model.S for j in
            model.S if i < j
        ) + sum(
            norm([model.w[(i, j), d] for d in model.D], [model.w[(j, i), d] for d in model.D]) for i in model.P for j in
            model.S
        )

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    eta = {
        i: min(norm(terminals[i], terminals[j]) for j in model.P if i != j)
        for i in model.P
    }

    def geometric_cut_50(model, i, j, s):
        # i, j = edge
        a = 0
        if norm(terminals[i], terminals[j]) > (eta[i] ** 2 + eta[j] ** 2 + eta[i] * eta[j]) ** 0.5:
            return model.y[i, s] + model.y[j, s] <= 1
        else:
            return pyo.Constraint.Skip

    if use_geometric_cut_50:
        model.geometric_cut_50 = pyo.Constraint(model.P, model.P, model.S, rule=geometric_cut_50)

    return model


if __name__ == '__main__':
    m = dbt_relaxed_alpha0(
        [[0, 1], [0, 0], [1, 0], [1, 1]],
        masses=[0, 0, 0, 0],
        alpha=0,
        relax_w=False,
        relax_y=False,
        disjunctive_w=2,
        use_geometric_cut_50=False
    )

    m.pprint()
