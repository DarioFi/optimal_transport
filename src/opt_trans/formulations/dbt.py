import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def norm(p1, p2):
    return sum((x - y) ** 2 for x, y in zip(p1, p2)) ** 0.5


def norm_y(p1, p2, edge):
    # assert abs(edge) < 1e-8 or abs(edge - 1) < 1e-8
    return sum((x * edge - y * edge) ** 2 for x, y in zip(p1, p2)) ** 0.5


def dbt(terminals, masses, alpha, *, use_bind_first_steiner, use_obj_lb, use_convex_hull, use_better_obj):

    """
    this corresponds to the first DBT formulation, without quadratic constraints
    :param terminals: list of list representing terminals coordinates
    :param masses: demand / offer of each terminal
    :param alpha: cost parameter
    :param use_bind_first_steiner: whether to fix the first steiner point to the first terminal
    :param use_obj_lb: Not Implemented
    :param use_convex_hull: Whether to use convex hull constraint
    :param use_better_obj: Whether  to use the better objective function
    :param use_gurobi: Whether to introduce auxiliary variables to accomodate GUROBI 11 limitation
    :return:
    :return:
    """

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
        return sum(model.y[k, j] for (k, j) in model.E if i in (k, j)) == 3

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

        # Write a BARON options file to specify RELAXATION_ONLY_EQUATIONS
        with open('baron_options.txt', 'w') as f:
            f.write('RELAXATION_ONLY_EQUATIONS {}\n'.format(' '.join("convex_hull_constraint")))
            f.write('RELAXATION_ONLY_EQUATIONS {}\n'.format(' '.join("convex_hull_sum_constraint")))

    def objective_rule(model):

        if use_better_obj:
            return (sum(
                model.f[i, j] ** alpha * norm_y(terminals[i], [model.x[j, d] for d in model.D], model.y[i, j]) for
                (i, j) in model.E1) +
                    sum(
                        model.f[i, j] ** alpha * norm_y([model.x[i, d] for d in model.D],
                                                        [model.x[j, d] for d in model.D],
                                                        model.y[i, j]) for
                        (i, j) in model.E2))

        else:
            return sum(
                model.f[i, j] ** alpha * norm(terminals[i], [model.x[j, d] for d in model.D]) for (i, j) in
                model.E1) + sum(
                model.f[i, j] ** alpha * norm([model.x[i, d] for d in model.D], [model.x[j, d] for d in model.D]) for
                (i, j)
                in model.E2)

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    return model


def dbt_alpha_0(terminals, masses, alpha, *, use_bind_first_steiner, use_obj_lb, use_convex_hull, use_better_obj,
                use_gurobi=False):
    """
    this corresponds to the first DBT formulation, without quadratic constraints and without flows (alpha=0)
    :param terminals: list of list representing terminals coordinates
    :param masses: demand / offer of each terminal
    :param alpha: cost parameter
    :param use_bind_first_steiner: whether to fix the first steiner point to the first terminal
    :param use_obj_lb: Not Implemented
    :param use_convex_hull: Whether to use convex hull constraint
    :param use_better_obj: Whether  to use the better objective function
    :param use_gurobi: Whether to introduce auxiliary variables to accomodate GUROBI 11 limitation
    :return:
    """


    assert len(terminals) == len(masses)
    assert abs(sum(masses)) < 1e-7
    assert masses[0] < 0
    assert all(m > 0 for m in masses[1:])  # masses are irrelevant
    assert alpha == 0

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

    if use_gurobi:
        model.x = pyo.Var(model.S.union(model.P), model.D, domain=pyo.Reals, bounds=(0, 1))
    else:
        model.x = pyo.Var(model.S, model.D, domain=pyo.Reals, bounds=(0, 1))

    model.y = pyo.Var(model.E, domain=pyo.Binary)

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

    def degree_constraint(model, i):
        return sum(model.y[k, j] for (k, j) in model.E if i in (k, j)) == 3

    model.degree_constraint = pyo.Constraint(model.S, rule=degree_constraint)

    if use_obj_lb:
        raise NotImplementedError("Objective lower bound not implemented yet")

    if use_bind_first_steiner:
        # bind the source to the first steiner point
        model.y[0, min(model.S)].fix(1)

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

    max_norm = D ** 0.5

    if use_gurobi:
        model.norm = pyo.Var(model.E, domain=pyo.NonNegativeReals, bounds=(0, max_norm))

        # def x_equality_constraint(model, i, d):
        #     return model.x[i, d] == terminals[i][d]
        # model.x_equality_constraint = pyo.Constraint(model.P, model.D, rule=x_equality_constraint)

        for i in model.P:
            for d in model.D:
                model.x[i, d].fix(terminals[i][d])

        # model.x[3, 0].fix(.5)
        # model.x[3, 1].fix(.7)z

        if use_better_obj:

            model.inside_norm = pyo.Var(model.E, model.D, domain=pyo.NonNegativeReals, bounds=(0, 1))
            model.inside_norm_not_sq = pyo.Var(model.E, model.D, domain=pyo.NonNegativeReals, bounds=(0, 1))

            def inside_norm_constraint(model, i, j, d):
                return model.inside_norm_not_sq[i, j, d] >= (
                        model.x[i, d] * model.y[i, j] - model.x[j, d] * model.y[i, j])

            def inside_norm_sq_constraint(model, i, j, d):
                return model.inside_norm[i, j, d] >= model.inside_norm_not_sq[i, j, d] ** 2

            model.inside_norm_sq_constraint = pyo.Constraint(model.E, model.D, rule=inside_norm_sq_constraint)
            model.inside_norm_constraint = pyo.Constraint(model.E, model.D, rule=inside_norm_constraint)


        else:
            model.inside_norm = pyo.Var(model.E, model.D, domain=pyo.NonNegativeReals, bounds=(0, 1))

            def inside_norm_constraint(model, i, j, d):
                return model.inside_norm[i, j, d] >= (model.x[i, d] - model.x[j, d]) ** 2

            model.inside_norm_constraint = pyo.Constraint(model.E, model.D, rule=inside_norm_constraint)

        def norm_e2_constraint(model, i, j):
            return model.norm[i, j] ** 2 >= sum(model.inside_norm[i, j, d] for d in model.D)

        model.norm_constraint = pyo.Constraint(model.E, rule=norm_e2_constraint)

        def objective_rule(model):
            return sum(model.y[i, j] * model.norm[i, j] for (i, j) in model.E)

        model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)


    else:
        def objective_rule(model):

            if use_better_obj:
                return (sum(
                    model.y[i, j] *
                    norm_y(
                        terminals[i], [model.x[j, d] for d in model.D], model.y[i, j]) for (i, j) in model.E1) +
                        sum(model.y[i, j] * norm_y([model.x[i, d] for d in model.D],
                                                   [model.x[j, d] for d in model.D],
                                                   model.y[i, j]) for (i, j) in model.E2))

            else:
                return sum(

                    model.y[i, j] * norm(terminals[i], [model.x[j, d] for d in model.D]) for (i, j) in
                    model.E1) + sum(
                    model.y[i, j] * norm([model.x[i, d] for d in model.D], [model.x[j, d] for d in model.D]) for
                    (i, j)
                    in model.E2)

        model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)



    return model


if __name__ == '__main__':

    # example usage:

    terminals = [(0, 1), (1, 0), (1, 1), (0, 0), (3, 5), (10, 10)]
    masses = [-1, .2, .3, .2, .2, .1]
    maximum_degree = 3
    alpha = .5

    bind_first_steiner = False
    use_obj_lb = False
    use_convex_hull = False

    for use_better_obj in [True, False]:
        model = dbt(terminals, masses, alpha, use_bind_first_steiner=bind_first_steiner, use_obj_lb=use_obj_lb,
                    use_convex_hull=use_convex_hull,
                    use_better_obj=use_better_obj)

        solver = SolverFactory('baron')
        results = solver.solve(model, tee=False)
        print(f"------{use_better_obj=}------")
        print(results.problem.lower_bound, results.problem.upper_bound)
        print(results.problem.cpu_time)
        print(results.problem.wall_time)
