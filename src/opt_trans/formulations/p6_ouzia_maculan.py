import pyomo.environ as pyo

from opt_trans.problems.utils import norm


def create_p6_model(terminals, steiner_points):
    """
    Implementation of a formulaiton (P6) from a paper of Ouzia and Maculan
    :param terminals:
    :param steiner_points:
    :return:
    """
    model = pyo.ConcreteModel()

    P = len(terminals)
    S = len(steiner_points)
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

    # fix model.x
    # model.x[3,0].fix(.3)
    # model.x[3,1].fix(.3)

    # objective minimize sum od d_i_j
    model.Obj = pyo.Objective(expr=sum(model.ds[i, j] for i in model.S.union(model.P) for j in model.S if i < j),
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
        return sum(model.y_p_s[i, j] for i in model.P) + sum(model.y_s_s[k, j] for k in model.S) == 3

    model.steiner_degree_constraint = pyo.Constraint(model.S, rule=steiner_degree_constraint)

    def steiner_connectivity_constraint(model, j):
        if j == P:
            return pyo.Constraint.Skip
        return sum(model.y_s_s[k, j] for k in model.S if k < j) == 1

    model.steiner_connectivity_constraint = pyo.Constraint(model.S, rule=steiner_connectivity_constraint)

    # This loses performance
    def steiner_connectivity_constraint(model, j):
        return sum(model.y_p_s[i, j] for i in model.P) <= 2

    # model.steiner_connectivity_constraint = pyo.Constraint(model.S, rule=steiner_connectivity_constraint)

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

    # add constraints for ds
    def ds_constraint(model, i, j):
        if i == j:
            return pyo.Constraint.Skip
        return model.ds[i, j] ** 2 >= sum(model.t[i, j, k] ** 2 for k in model.D)

    model.ds_constraint = pyo.Constraint(model.S.union(model.P), model.S, rule=ds_constraint)

    # add constraints for t
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
