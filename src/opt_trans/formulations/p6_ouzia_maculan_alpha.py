import pyomo.environ as pyo


def norm(p1, p2, dim):
    return sum((p1[d] - p2[d]) ** 2 for d in dim) ** 0.5


def p6_model_alpha(terminals, terminal_masses, alpha):
    """
    Implementation of a formulaiton (P6) from a paper of Ouzia and Maculan with extra flow constraints to accommodate flows
    :param terminals:
    :param steiner_points:
    :return:
    """
    model = pyo.ConcreteModel()

    P = len(terminals)
    S = len(terminals) - 2
    D = len(terminals[0])  # Dimension of points

    # Sets
    model.P = pyo.RangeSet(0, P - 1)
    model.S = pyo.RangeSet(P, P + S - 1)
    model.D = pyo.RangeSet(0, D - 1)

    # Variables

    model.y_p_s = pyo.Var(model.P, model.S, domain=pyo.Binary)
    model.y_s_s = pyo.Var(model.S, model.S, domain=pyo.Binary)

    model.ds = pyo.Var(model.S.union(model.P), model.S, domain=pyo.NonNegativeReals)

    model.t = pyo.Var(model.S.union(model.P), model.S, model.D, domain=pyo.Reals)

    model.x = pyo.Var(model.S, model.D, domain=pyo.Reals)

    model.f = pyo.Var(model.S.union(model.P), model.S.union(model.P), domain=pyo.Reals)

    model.abs_f = pyo.Var(model.S.union(model.P), model.S.union(model.P), domain=pyo.NonNegativeReals)

    # objective minimize sum od d_i_j
    model.Obj = pyo.Objective(
        expr=sum(model.ds[i, j] * model.abs_f[i, j] ** alpha for i in model.S.union(model.P) for j in model.S if i < j)
             + 0.001 * sum(model.abs_f[i, j] for i in model.S.union(model.P) for j in model.S.union(model.P))
        ,
        sense=pyo.minimize)

    def y_s_s_constraint(model, i, j):
        if i < j:
            return model.y_s_s[i, j] == model.y_s_s[j, i]
        return pyo.Constraint.Skip

    model.y_s_s_constraint = pyo.Constraint(model.S, model.S, rule=y_s_s_constraint)

    def y_s_s_diagonal_constraint(model, i):
        return model.y_s_s[i, i] == 0

    model.y_s_s_diagonal_constraint = pyo.Constraint(model.S, rule=y_s_s_diagonal_constraint)

    def terminal_degree_constraint(model, i):
        return sum(model.y_p_s[i, j] for j in model.S) == 1

    model.terminal_degree_constraint = pyo.Constraint(model.P, rule=terminal_degree_constraint)

    def steiner_degree_constraint(model, j):
        return sum(model.y_p_s[i, j] for i in model.P) + sum(model.y_s_s[k, j] for k in model.S) <= 3

    model.steiner_degree_constraint = pyo.Constraint(model.S, rule=steiner_degree_constraint)

    def steiner_connectivity_constraint(model, j):
        if j == P:
            return pyo.Constraint.Skip
        return sum(model.y_s_s[k, j] for k in model.S if k < j) == 1

    model.steiner_connectivity_constraint = pyo.Constraint(model.S, rule=steiner_connectivity_constraint)

    # This loses performance
    def steiner_connectivity_constraint(model, j):
        return sum(model.y_p_s[i, j] for i in model.P) <= 2

    eta = {
        i: min(norm(terminals[i], terminals[j], model.D) for j in model.P if i != j)
        for i in model.P
    }

    print(eta)

    def geometric_cut_50(model, i, j, s):
        if i >= j:
            return pyo.Constraint.Skip
        if norm(terminals[i], terminals[j], model.D) > (
                eta[i] ** 2 + eta[j] ** 2 + eta[i] * eta[j]) ** .5:
            return model.y_p_s[i, s] + model.y_p_s[j, s] <= 1
        return pyo.Constraint.Skip

    model.geometric_cut_50 = pyo.Constraint(model.P, model.P, model.S, rule=geometric_cut_50)

    def ds_constraint(model, i, j):
        if i == j:
            return pyo.Constraint.Skip
        return model.ds[i, j] ** 2 >= sum(model.t[i, j, k] ** 2 for k in model.D)

    model.ds_constraint = pyo.Constraint(model.S.union(model.P), model.S, rule=ds_constraint)

    def t_constraint_42_part1(model, i, j, k):
        if i == j:
            return pyo.Constraint.Skip
        if i in model.P:
            return (- model.y_p_s[i, j]) <= model.t[i, j, k]
        else:
            return (- model.y_s_s[i, j]) <= model.t[i, j, k]

    def t_constraint_42_part2(model, i, j, k):
        if i == j:
            return pyo.Constraint.Skip
        if i in model.P:
            return model.t[i, j, k] <= model.y_p_s[i, j]
        else:
            return model.t[i, j, k] <= model.y_s_s[i, j]

    def t_constraint_43_part1(model, i, j, k):
        if i == j:
            return pyo.Constraint.Skip
        if i in model.P:
            return -(1 - model.y_p_s[i, j]) + (terminals[i][k] - model.x[j, k]) <= model.t[i, j, k]

        else:
            return -(1 - model.y_s_s[i, j]) + (model.x[i, k] - model.x[j, k]) <= model.t[i, j, k]

    def t_constraint_43_part2(model, i, j, k):
        if i == j:
            return pyo.Constraint.Skip
        if i in model.P:
            return model.t[i, j, k] <= 1 - model.y_p_s[i, j] + (terminals[i][k] - model.x[j, k])

        else:
            return model.t[i, j, k] <= 1 - model.y_s_s[i, j] + (model.x[i, k] - model.x[j, k])

    model.t_constraint_42_part1 = pyo.Constraint(model.S.union(model.P), model.S, model.D, rule=t_constraint_42_part1)
    model.t_constraint_42_part2 = pyo.Constraint(model.S.union(model.P), model.S, model.D, rule=t_constraint_42_part2)
    model.t_constraint_43_part1 = pyo.Constraint(model.S.union(model.P), model.S, model.D, rule=t_constraint_43_part1)
    model.t_constraint_43_part2 = pyo.Constraint(model.S.union(model.P), model.S, model.D, rule=t_constraint_43_part2)

    # flow section

    # reflection of f
    def f_reflection_constraint(model, i, j):
        if i in model.P and j in model.P:
            return model.f_sum_normalized[i, j] == 0
        if i != j:
            return model.f_sum_normalized[i, j] == - model.f_sum_normalized[j, i]
        else:
            return model.f_sum_normalized[i, j] == 0

    model.f_reflection_constraint = pyo.Constraint(model.S.union(model.P), model.S.union(model.P),
                                                   rule=f_reflection_constraint)

    # absolute value of f
    def abs_f_constraint_part1(model, i, j):
        return model.abs_f[i, j] >= model.f_sum_normalized[i, j]

    def abs_f_constraint_part2(model, i, j):
        return model.abs_f[i, j] >= - model.f_sum_normalized[i, j]

    model.abs_f_constraint_part1 = pyo.Constraint(model.S.union(model.P), model.S.union(model.P),
                                                  rule=abs_f_constraint_part1)
    model.abs_f_constraint_part2 = pyo.Constraint(model.S.union(model.P), model.S.union(model.P),
                                                  rule=abs_f_constraint_part2)

    # flow conservation
    def flow_conservation_steiners(model, i):
        return sum(model.f_sum_normalized[i, j] for j in model.S.union(model.P)) == 0

    def flow_conservation_terminals(model, i):
        return sum(model.f_sum_normalized[i, j] for j in model.S.union(model.P)) == terminal_masses[i]

    model.flow_conservation_steiners = pyo.Constraint(model.S, rule=flow_conservation_steiners)
    model.flow_conservation_terminals = pyo.Constraint(model.P, rule=flow_conservation_terminals)

    # flow active only if y_p_s or y_s_s is active
    def flow_active_constraint_p_1(model, i, j):
        return model.f_sum_normalized[i, j] <= model.y_p_s[i, j]

    def flow_active_constraint_p_2(model, i, j):
        return model.f_sum_normalized[i, j] >= - model.y_p_s[i, j]

    def flow_active_constraint_s_1(model, i, j):
        return model.f_sum_normalized[i, j] <= model.y_s_s[i, j]

    def flow_active_constraint_s_2(model, i, j):
        return model.f_sum_normalized[i, j] >= - model.y_s_s[i, j]

    model.flow_active_constraint_p_1 = pyo.Constraint(model.P, model.S, rule=flow_active_constraint_p_1)
    model.flow_active_constraint_p_2 = pyo.Constraint(model.P, model.S, rule=flow_active_constraint_p_2)
    model.flow_active_constraint_s_1 = pyo.Constraint(model.S, model.S, rule=flow_active_constraint_s_1)
    model.flow_active_constraint_s_2 = pyo.Constraint(model.S, model.S, rule=flow_active_constraint_s_2)

    # get convex hull vertices for terminals
    import scipy.spatial as sp
    hull = sp.ConvexHull(terminals)
    A = hull.equations[:, :-1]
    b = -hull.equations[:, -1]

    # Add convex hull constraints for model.x
    def convex_hull_constraint(model, s, i):
        return sum(A[i, d] * model.x[s, d] for d in model.D) <= b[i]

    # convex hull as convex combinations of terminals
    model.a = pyo.Var(model.P, model.S, domain=pyo.NonNegativeReals)

    def convex_hull_combinations_constraint(model, s, k):
        return sum(model.a[i, s] * terminals[i][k] for i in model.P) == model.x[s, k]

    # model.convex_hull_constraint = pyo.Constraint(model.S, range(len(b)), rule=convex_hull_constraint)
    model.convex_combinations = pyo.Constraint(model.S, model.D, rule=convex_hull_combinations_constraint)

    model.pprint()
    return model
