import pyomo.environ as pyo


def norm(p1, p2):
    return sum((x - y) ** 2 for x, y in zip(p1, p2)) ** 0.5


def dbt_relaxed_alpha0(terminals, alpha, masses, relax_y: bool, relax_w: bool):
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

    model.x = pyo.Var(model.S, model.D, domain=pyo.Reals, bounds=(0, 1))

    # todo: relax y to reals?
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

    w_index = [(i, j) for i in model.S for j in model.S if i != j] + [(i, j) for i in model.S for j in model.P] + [(j, i) for i in
                                                                                                         model.S for j
                                                                                                         in model.P]

    model.w = pyo.Var(w_index, model.D, domain=pyo.NonNegativeReals, bounds=(0, 1))

    if not relax_w:
        def w_s_s_constraint(model, i, j, d):
            if i < j:
                return model.w[(i, j), d] == model.x[i, d] * model.y[i, j]
            elif i > j:
                return model.w[(i, j), d] == model.x[i, d] * model.y[j, i]
            else:
                return pyo.Constraint.Skip

        model.w_s_s_cosntraint = pyo.Constraint(model.S, model.S, model.D, rule=w_s_s_constraint)

        def w_p_s_constraint(model, i, j, d):
            if i in model.P:
                return model.w[(i, j), d] == model.y[i, j] * terminals[i][d]
            elif j in model.P:
                return model.w[(i, j), d] == model.y[j, i] * model.x[i, d]

        model.w_p_s_constraint = pyo.Constraint(model.S, model.P, model.D, rule=w_p_s_constraint)
        model.w_s_p_constraint = pyo.Constraint(model.P, model.S, model.D, rule=w_p_s_constraint)

    else:

        # instead of having w = x * y we now need
        # w >= 0
        # w >= x + y - 1
        # w <= x
        # w <= y

        def w_relaxation_constraint_1(model, i, j, d):
            if i == j: return pyo.Constraint.Skip
            return model.w[(i, j), d] >= 0

        def w_relaxation_constraint_2(model, i, j, d):
            if i < j:
                return model.w[(i, j), d] >= model.x[i, d] + model.y[i, j] - 1
            elif j < i:
                return model.w[(i, j), d] >= model.x[i, d] + model.y[j, i] - 1
            else:
                return pyo.Constraint.Skip

        def w_relaxation_constraint_3(model, i, j, d):
            if i == j: return pyo.Constraint.Skip
            return model.w[(i, j), d] <= model.x[i, d]

        def w_relaxation_constraint_4(model, i, j, d):

            if i < j:
                return model.w[(i, j), d] <= model.y[i, j]
            elif j < i:
                return model.w[(i, j), d] <= model.y[j, i]
            else:
                return pyo.Constraint.Skip

        model.w_relaxation_constraint_1 = pyo.Constraint(model.S, model.S, model.D, rule=w_relaxation_constraint_1)
        model.w_relaxation_constraint_2 = pyo.Constraint(model.S, model.S, model.D, rule=w_relaxation_constraint_2)
        model.w_relaxation_constraint_3 = pyo.Constraint(model.S, model.S, model.D, rule=w_relaxation_constraint_3)
        model.w_relaxation_constraint_4 = pyo.Constraint(model.S, model.S, model.D, rule=w_relaxation_constraint_4)

        model.w_relaxation_constraint_1_p = pyo.Constraint(model.S, model.P, model.D, rule=w_relaxation_constraint_1)
        model.w_relaxation_constraint_2_p = pyo.Constraint(model.S, model.P, model.D, rule=w_relaxation_constraint_2)
        model.w_relaxation_constraint_3_p = pyo.Constraint(model.S, model.P, model.D, rule=w_relaxation_constraint_3)
        model.w_relaxation_constraint_4_p = pyo.Constraint(model.S, model.P, model.D, rule=w_relaxation_constraint_4)

        # w[p, s] does not need to be relaxed as w = terminal * y is alread a linear constraint
        # implement this

        def w_p_s_non_relaxed(model, i, j, d):
            return model.w[(i, j), d] == model.y[i, j] * terminals[i][d]

        model.w_p_s_non_relaxed = pyo.Constraint(model.P, model.S, model.D, rule=w_p_s_non_relaxed)

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
        relax_y=True
    )

    m.pprint()
