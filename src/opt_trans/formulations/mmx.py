import math
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
from random import random


def euclidean_distance_pyomo(p1, p2, dim):
    return sum((p1[d] - p2[d]) ** 2 for d in dim) ** 0.5


def mmx_model(terminals, use_new_objective, use_convex_hull, use_convex_combinations, use_geometric_cuts):
    """
    MMX model
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

    # Create binary variables y_{ij}
    model.y_p_s = pyo.Var(model.P, model.S, domain=pyo.Binary)
    model.y_s_s = pyo.Var(model.S, model.S, domain=pyo.Binary)

    def y_s_s_constraint(model, i, j):
        if i < j:
            return model.y_s_s[i, j] == model.y_s_s[j, i]
        return pyo.Constraint.Skip

    model.y_s_s_constraint = pyo.Constraint(model.S, model.S, rule=y_s_s_constraint)

    # fix y_s_s[i,i] = 0
    def y_s_s_diagonal_constraint(model, i):
        return model.y_s_s[i, i] == 0

    model.y_s_s_diagonal_constraint = pyo.Constraint(model.S, rule=y_s_s_diagonal_constraint)

    # Create continuous variables x^j for Steiner points
    model.x = pyo.Var(model.S, model.D, domain=pyo.Reals)

    if not use_new_objective:
        # Objective function: minimize the weighted sum of distances
        model.Obj = pyo.Objective(
            expr=sum(
                model.y_p_s[i, j] * euclidean_distance_pyomo(terminals[i], [model.x[j, d] for d in model.D], model.D)
                for i in model.P
                for j in model.S
            ) + sum(
                model.y_s_s[i, j] * euclidean_distance_pyomo([model.x[i, d] for d in model.D],
                                                             [model.x[j, d] for d in model.D], model.D)
                for i in model.S
                for j in model.S
                if i < j
            ),
            sense=pyo.minimize  # Assuming you want to minimize the objective
        )
    else:
        model.Obj = pyo.Objective(
            expr=sum(
                sum((terminals[i][d] * model.y_p_s[i, j] - model.x[j, d] * model.y_p_s[i, j]) ** 2 for d in
                    model.D) ** .5
                for i in model.P
                for j in model.S
            ) + sum(
                sum((model.x[i, d] * model.y_s_s[i, j] - model.x[j, d] * model.y_s_s[i, j]) ** 2 for d in model.D) ** .5
                for i in model.S
                for j in model.S
                if i < j
            ),
            sense=pyo.minimize  # Assuming you want to minimize the objective
        )

    model.Obj.sense = pyo.minimize

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

    if use_geometric_cuts:
        eta = {
            i: min(euclidean_distance_pyomo(terminals[i], terminals[j], model.D) for j in model.P if i != j)
            for i in model.P
        }

        def geometric_cut_50(model, i, j, s):
            if i >= j:
                return pyo.Constraint.Skip
            if euclidean_distance_pyomo(terminals[i], terminals[j], model.D) > (
                    eta[i] ** 2 + eta[j] ** 2 + eta[i] * eta[j]) ** .5:
                return model.y_p_s[i, s] + model.y_p_s[j, s] <= 1
            return pyo.Constraint.Skip

        model.geometric_cut_50 = pyo.Constraint(model.P, model.P, model.S, rule=geometric_cut_50)

    if use_convex_hull:
        import scipy.spatial as sp
        hull = sp.ConvexHull(terminals)
        A = hull.equations[:, :-1]
        b = -hull.equations[:, -1]

        # Add convex hull constraints for model.x
        def convex_hull_constraint(model, s, i):
            return sum(A[i, d] * model.x[s, d] for d in model.D) <= b[i]

        model.convex_hull_constraint = pyo.Constraint(model.S, range(len(b)), rule=convex_hull_constraint)

    if use_convex_combinations:
        def convex_hull_combinations_constraint(model, s, k):
            return sum(model.a[i, s] * terminals[i][k] for i in model.P) == model.x[s, k]

        model.a = pyo.Var(model.P, model.S, domain=pyo.NonNegativeReals)
        model.convex_combinations = pyo.Constraint(model.S, model.D, rule=convex_hull_combinations_constraint)

    return model


