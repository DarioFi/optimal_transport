import itertools
import math
import random
from typing import Dict

import pyomo.environ as pyo

from opt_trans.formulations.utils import get_instance_specific_bound, get_general_cosine_bound
from opt_trans.problems.instance_generators import random_points_unit_square_with_masses


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
    Add constraint z[i_z] = x[i_x] * y[i_y] for (i_z, i_x, i_y) in index_sets.
    Serves as a utility for building the DBTQ formulation and reduce duplciated code
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
    Utility for disjunctive w formulation
    :return:
    """
    fs = [{
        a: b for a in A for b in b_assignemnt
    } for b_assignemnt in itertools.product(B, repeat=len(A))]
    return [pyomo_wrapper(f) for f in fs]


def dbtq(terminals, alpha, masses, relax_y: bool, relax_w: bool, disjunctive_w: int | bool,
         use_geometric_cut_50: bool, angles_constraint: bool, starting_position: Dict = None, cosine_upper_bound=None):
    """
    Implements the DBTQ formulation at alpha=0 with the features described in the thesis
    :return:
    """
    assert alpha == 0
    P = len(terminals)
    S = len(terminals) - 2
    D = len(terminals[0])  # Dimension of points

    # assert terminals are in the unit cube
    for i in range(P):
        for d in range(D):
            assert terminals[i][d] >= 0 and terminals[i][d] <= 1

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

    # endregion

    # region bilinear-terms

    # some w are unaffected by the disjunctive relaxation, those who do not depend on disjunctive_s
    # in particular those which depend on terminals are always unaffected
    # the rest are affected and need to be split into two parts

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

    add_bilinear(model, model.w, model.x, model.y, w_index_s_p_classic, relax_w, "w_bilinear_constraint_s_p")

    if disjunctive_w:
        w_index_disjunctive = [
            ((i, j), d) for i in disjunctive_s for j in model.S.union(model.P) for d in model.D
            if i != j
        ]

        w_index_disjunctive_wrap = [pyomo_wrapper(x) for x in w_index_disjunctive]

        w_index_disjunctive_2 = [(x, d) for x in w_index for d in model.D if x[0] in disjunctive_s]

        assert sorted(list(w_index_disjunctive)) == sorted(list(w_index_disjunctive_2))

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

    # region angles

    # (x[i] - x[j], x[k] - x[j]) = -1/2 |x[i] - x[j]| | |x[k] - x[j]|
    # warning: this holds when the i,j,k are in the correct order and only if they are connected

    if angles_constraint:

        if cosine_upper_bound is None:
            raise ValueError(
                "For security reasons, cosine_upper_bound must be provided. DBTQ assumes alpha=0 and gets called"
                "as a subroutine of dbtq_with_flows, which provides it.")

        M_val = 10

        model.epsilon = 1e-4

        # Auxiliary variable for absolute value |w - p|
        model.delta = pyo.Var(model.S, model.P, model.D, domain=pyo.NonNegativeReals)

        # Auxiliary variable for L1 distance dist(i,j)
        model.dist = pyo.Var(model.S, model.P, domain=pyo.NonNegativeReals)

        # Auxiliary binary variable b[i,j] = 1 iff dist[i,j] <= epsilon
        model.b = pyo.Var(model.S, model.P, domain=pyo.Binary)

        model.x_active = pyo.Var(model.S, domain=pyo.Binary)

        # --- 3. Define Constraints ---

        # Constraints to linearize absolute value: delta[i,j,d] >= |w[i,d] - p[j,d]|
        def abs_val_pos_rule(m, i, j, d):
            return m.delta[i, j, d] >= m.x[i, d] - terminals[j][d]

        model.abs_val_pos_con = pyo.Constraint(model.S, model.P, model.D, rule=abs_val_pos_rule)

        def abs_val_neg_rule(m, i, j, d):
            return m.delta[i, j, d] >= -(m.x[i, d] - terminals[j][d])

        model.abs_val_neg_con = pyo.Constraint(model.S, model.P, model.D, rule=abs_val_neg_rule)

        # Constraint to define L1 distance: dist[i,j] = sum(delta[i,j,d])
        # In optimization, if the objective minimizes distance, >= might suffice.
        # But = is safer
        def dist_calc_rule(m, i, j):
            return m.dist[i, j] == sum(m.delta[i, j, d] for d in m.D)

        model.dist_calc_con = pyo.Constraint(model.S, model.P, rule=dist_calc_rule)

        # Constraints to link dist[i,j] and binary indicator b[i,j]
        def indicator_upper_bound_rule(m, i, j):
            return m.dist[i, j] <= m.epsilon + M_val * (1 - m.b[i, j])

        model.Sndicator_upper_con = pyo.Constraint(model.S, model.P, rule=indicator_upper_bound_rule)

        # Constraint 2: b[i,j] = 0 => dist[i,j] > epsilon (or >= epsilon + small_delta)
        def indicator_lower_bound_rule(m, i, j):
            return m.dist[i, j] >= m.epsilon * (1 - m.b[i, j])

        model.Sndicator_lower_con = pyo.Constraint(model.S, model.P, rule=indicator_lower_bound_rule)

        # Constraints to link y[i] and b[i,j]
        def y_link_lower_rule(m, i):
            return sum(m.b[i, j] for j in m.P) >= m.x_active[i]

        model.y_link_lower_con = pyo.Constraint(model.S, rule=y_link_lower_rule)

        # Constraint 2: y[i] = 0 => sum(b[i,j]) = 0
        # Implemented using Big M: sum(b[i,j]) <= |J| * y[i]
        def y_link_upper_rule(m, i):
            return sum(m.b[i, j] for j in m.P) <= len(m.P) * m.x_active[i]

        model.y_link_upper_con = pyo.Constraint(model.S, rule=y_link_upper_rule)

        def angle_constraint(model, i, j, k):
            return model.x_active[j] * sum(
                (model.w[(i, j), d] - model.w[(j, i), d]) * (model.w[(k, j), d] - model.w[(j, k), d])
                for d in model.D
            ) <= \
                model.x_active[j] * cosine_upper_bound * norm([model.w[(i, j), d] for d in model.D],
                                                              [model.w[(j, i), d] for d in model.D]) * norm(
                    [model.w[(k, j), d] for d in model.D], [model.w[(j, k), d] for d in model.D])

        use = model.S
        index_angles = [
            (i, j, k) for i in use for j in use for k in use if i != j and i != k and j != k and i < k
        ]
        model.angle_constraint = pyo.Constraint(index_angles, rule=angle_constraint)

    # endregion

    def objective_rule(model):
        return sum(
            norm([model.w[(i, j), d] for d in model.D], [model.w[(j, i), d] for d in model.D]) for i in model.S for
            j in
            model.S if i < j
        ) + sum(
            norm([model.w[(i, j), d] for d in model.D], [model.w[(j, i), d] for d in model.D]) for i in model.P for
            j in
            model.S
        )

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    eta = {
        i: min(norm(terminals[i], terminals[j]) for j in model.P if i != j)
        for i in model.P
    }

    def geometric_cut_50(model, i, j, s):
        a = 0
        if norm(terminals[i], terminals[j]) > (eta[i] ** 2 + eta[j] ** 2 + eta[i] * eta[j]) ** 0.5:
            return model.y[i, s] + model.y[j, s] <= 1
        else:
            return pyo.Constraint.Skip

    if use_geometric_cut_50:
        model.geometric_cut_50 = pyo.Constraint(model.P, model.P, model.S, rule=geometric_cut_50)

    if starting_position is not None:
        for key, value in starting_position.items():
            try:
                var = getattr(model, key)

                for i, v in value.items():
                    if isinstance(i, str):
                        i = eval(i)
                    var[i].value = v

            except AttributeError:
                raise AttributeError(f"Model does not have attribute {key}")

    return model


def dbtq_with_flows(terminals, alpha, masses, relax_y: bool, relax_w: bool, disjunctive_w: int | bool,
                    use_geometric_cut_50: bool, angles_constraint: bool, starting_position: Dict = None,
                    use_log_obj=False, log_multiplier=None, cosine_upper_bound=None):
    """
    Implements the DBTQ formulation with flow variables and constraints. It builds on top of the dbtq without flows formulation,
    passing most arguments to it.
    It also supports logarithmic objective function.
    """

    assert use_geometric_cut_50 is False, "Geometric cut 50 is not supported in this version as it is not guaranteed to be optimal"

    if cosine_upper_bound is None:
        cosine_upper_bound = get_instance_specific_bound(masses, alpha)
    if cosine_upper_bound == "general":
        cosine_upper_bound = get_general_cosine_bound(alpha)

    model = dbtq(terminals, 0, masses, relax_y, relax_w, disjunctive_w,
                 use_geometric_cut_50, angles_constraint, starting_position, cosine_upper_bound=cosine_upper_bound)

    if alpha == 0:
        return model

    # flow variables
    first_edge = (0, min(model.S))
    model.f = pyo.Var(model.E, domain=pyo.NonNegativeReals, bounds=(0, 1))

    model.f[first_edge].fix(-masses[0])  # check that it is consistent

    # flow constraints

    def flow_conservation_steiner(model, i):
        inc = [(j, i) for j in model.S if j < i]
        out = [(i, j) for j in model.S if i < j] + [(j, i) for j in model.P if j != 0]
        if i in first_edge:
            inc += [first_edge]
        return sum(model.f[e] for e in inc) - sum(model.f[e] for e in out) == 0

    model.flow_conservation_steiner = pyo.Constraint(model.S, rule=flow_conservation_steiner)

    def flow_conservation_terminal(model, i):
        assert masses[i] > 0
        assert i != 0
        inc = [e for e in model.E if e[1] == i]
        out = [e for e in model.E if e[0] == i]
        assert len(inc) == 0 or len(out) == 0, "Terminal should not have both incoming and outgoing edges"
        return - sum(model.f[e] for e in inc) + sum(model.f[e] for e in out) == masses[i]

    # run over P without the first index
    P_w0 = [i for i in model.P if i != 0]
    model.flow_conservation_terminal = pyo.Constraint(P_w0, rule=flow_conservation_terminal)

    # y = 0 => f = 0
    def y_zero_f_zero(model, i, j):
        return model.f[i, j] <= model.y[i, j]

    model.y_zero_f_zero = pyo.Constraint(model.E, rule=y_zero_f_zero)

    # new objective function with flows
    del model.obj

    def objective_rule_alpha(model):
        return sum(
            model.f[(i, j)] ** alpha * norm([model.w[(i, j), d] for d in model.D],
                                            [model.w[(j, i), d] for d in model.D]) for i in model.S for
            j in
            model.S if i < j
        ) + sum(
            model.f[(i, j)] ** alpha * norm([model.w[(i, j), d] for d in model.D],
                                            [model.w[(j, i), d] for d in model.D]) for i in model.P for
            j in
            model.S
        )

    if use_log_obj:
        def objective_rule_alpha(model):
            const = math.log(log_multiplier + 1)
            return sum(
                pyo.log(log_multiplier * model.f[(i, j)] + 1) / const * norm([model.w[(i, j), d] for d in model.D],
                                                                             [model.w[(j, i), d] for d in model.D]) for
                i in model.S for
                j in
                model.S if i < j
            ) + sum(
                pyo.log(log_multiplier * model.f[(i, j)] + 1) / const * norm([model.w[(i, j), d] for d in model.D],
                                                                             [model.w[(j, i), d] for d in model.D]) for
                i in model.P for
                j in
                model.S
            )

    model.obj = pyo.Objective(rule=objective_rule_alpha, sense=pyo.minimize)

    return model

