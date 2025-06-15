import pyomo.environ as pyo
from opt_trans.formulations.utils import get_lower_bound

def norm(p1, p2, dim):
    return sum((p1[d] - p2[d]) ** 2 for d in dim) ** 0.5


def gmmx(terminals, masses, maximum_degree, alpha, use_bind_first_steiner, use_obj_lb, **kwargs):
    """
    Old suboptimal formulation for the DBT problem, based on generalizing the MMX formulation
    """

    assert len(terminals) == len(masses)
    assert abs(sum(masses)) < 1e-7
    model = pyo.ConcreteModel()

    P = len(terminals)
    S = len(terminals) - 2
    D = len(terminals[0])  # Dimension of points

    model.P = pyo.RangeSet(0, P - 1)
    model.S = pyo.RangeSet(P, P + S - 1)
    model.D = pyo.RangeSet(0, D - 1)

    # s union p
    model.SUP = pyo.RangeSet(0, P + S - 1)

    # we need:
    # - flows
    # - activations
    # - positions

    model.x = pyo.Var(model.S, model.D, domain=pyo.Reals)
    model.y = pyo.Var(model.SUP, model.SUP, domain=pyo.Binary)
    model.f = pyo.Var(model.SUP, model.SUP, domain=pyo.NonNegativeReals)

    # terminals have degree 1 constraint
    def degree_one_constraint(model, i):
        return sum(model.y[i, j] for j in model.SUP) == 1

    model.degree_one_constraint = pyo.Constraint(model.P, rule=degree_one_constraint)

    def connectivity_constraint(model, i):
        if i != min(model.S):
            return sum(model.y[i, j] for j in model.S if j < i) == 1
        return pyo.Constraint.Skip

    model.connectivity_constraint = pyo.Constraint(model.S, rule=connectivity_constraint)

    # flows constraint 0 on same node
    def flow_same_node_constraint(model, i):
        return model.f[i, i] == 0

    model.flow_same_node_constraint = pyo.Constraint(model.SUP, rule=flow_same_node_constraint)

    def activation_same_node_constraint(model, i):
        return model.y[i, i] == 0

    model.activation_same_node_constraint = pyo.Constraint(model.SUP, rule=activation_same_node_constraint)

    # conservation of flows at P
    def flow_conservation_constraint_P(model, i):
        return sum(model.f[i, j] for j in model.SUP) - sum(model.f[j, i] for j in model.SUP) == masses[i]

    model.flow_conservation_constraint_P = pyo.Constraint(model.P, rule=flow_conservation_constraint_P)

    # conservation of flows at S
    def flow_conservation_constraint_S(model, i):
        return sum(model.f[i, j] for j in model.SUP) - sum(model.f[j, i] for j in model.SUP) == 0

    model.flow_conservation_constraint_S = pyo.Constraint(model.S, rule=flow_conservation_constraint_S)

    # degree constraint at steiner points
    def degree_constraint_ub(model, i):
        return sum(model.y[i, j] for j in model.SUP) <= maximum_degree

    model.degree_constraint_ub = pyo.Constraint(model.S, rule=degree_constraint_ub)

    def degree_constraint_lb(model, i):
        return sum(model.y[i, j] for j in model.SUP) >= 3

    model.degree_constraint_lb = pyo.Constraint(model.S, rule=degree_constraint_lb)

    # flows smaller than y

    def flow_less_than_y_constraint(model, i, j):
        return model.f[i, j] <= model.y[i, j]

    model.flow_less_than_y_constraint = pyo.Constraint(model.SUP, model.SUP, rule=flow_less_than_y_constraint)

    def y_symmetric_constraint(model, i, j):
        if i < j:
            return model.y[i, j] == model.y[j, i]
        return pyo.Constraint.Skip
#  f_ij = 0 or f_ji = 0 in optimal solutions

    model.y_symmetric_constraint = pyo.Constraint(model.SUP, model.SUP, rule=y_symmetric_constraint)

    if use_bind_first_steiner:
        def single_constr(model, i, j):
            return model.y[i, j] == 1
        model.single_constr = pyo.Constraint([0], [P], rule=single_constr)

    # objective

    def objective_rule(model):

        return (
                sum((model.f[i, j] + model.f[j, i]) ** alpha * norm([model.x[i, d] for d in model.D],
                                                                [model.x[j, d] for d in model.D], model.D) for i in
                    model.S for j in
                    model.S if i < j) +
                sum((model.f[i, j] + model.f[j, i]) ** alpha * norm(terminals[i], [model.x[j, d] for d in model.D], model.D)
                    for i in model.P for j in
                    model.S)
        )

    if use_obj_lb:
        # it means that we use the bound that we know lb <= objective value
        lb = get_lower_bound(terminals, masses, alpha)

        def objective_lb_rule(model):
            return lb <= objective_rule(model)

        model.lb_obj = pyo.Constraint(rule=objective_lb_rule)



    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)


    return model

