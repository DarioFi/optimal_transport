from formulations.dbt import dbt_alpha_0, dbt


def baseline_polynomial_fixed_topology(terminals, masses, alpha, **kwargs):
    steiners = list(range(len(terminals), len(terminals) * 2 - 2))
    terminals_index = list(range(len(terminals)))

    # we build a binary tree with root steiner[0] and sorted steiners[1:]

    connections = [
        (terminals_index[0], steiners[0])
    ]

    print(connections)

    for count, i in enumerate(steiners):
        if count == 0:
            continue
        new_con = (steiners[(count - 1) // 2], i)
        print(new_con)
        connections.append(new_con)

    for c_count, i in enumerate(terminals_index[1:]):
        curr = c_count + count + 1
        new_con = (steiners[(curr - 1) // 2], i)
        print(new_con)
        connections.append(new_con)

    # compute flows bottom down

    flows = {}

    for i, j in reversed(connections):
        if j in terminals_index:
            flows[(i, j)] = masses[j]
        else:
            flows[(i, j)] = sum([flows[(j, k)] for k in steiners + terminals_index if (j, k) in flows])

    print(flows)

    # make new model now
    import pyomo.environ as pyo

    model = pyo.ConcreteModel()

    model.P = terminals_index
    model.S = steiners
    model.E = connections
    model.D = list(range(len(terminals[0])))

    model.flows = flows

    model.norms = pyo.Var(model.E, within=pyo.NonNegativeReals)
    model.x = pyo.Var(model.S, model.D, within=pyo.Reals)

    def norms_rule(model, i, j):
        if i in terminals_index:
            return model.norms[i, j] ** 2 == sum((terminals[i][d] - model.x[j, d]) ** 2 for d in model.D)

        if j in terminals_index:
            return model.norms[i, j] ** 2 == sum((model.x[i, d] - terminals[j][d]) ** 2 for d in model.D)

        return model.norms[i, j] ** 2 == sum((model.x[i, d] - model.x[j, d]) ** 2 for d in model.D)

    model.norms_rule = pyo.Constraint(model.E, rule=norms_rule)

    def obj_rule(model):
        return sum(model.flows[(i, j)] ** alpha * model.norms[i, j] for i, j in model.E)

    model.obj = pyo.Objective(rule=obj_rule)

    model.pprint()
    return model


if __name__ == '__main__':
    from problems.closest_counterexample import random_points_unit_square_with_masses

    alpha = 0.5
    data = random_points_unit_square_with_masses(7, alpha)

    terminals = data["terminals"]
    masses = data["masses"]

    print(masses)
    model = baseline_polynomial_fixed_topology(terminals, masses, alpha)

    from experiment import Experiment

    fs = dbt_alpha_0 if alpha == 0 else dbt
    for form in [baseline_polynomial_fixed_topology, fs]:
        exp = Experiment(
            formulation=form,
            solver="baron",
            solver_options="",
            instance_generator=random_points_unit_square_with_masses,
            instance_arguments={'n': 6, 'alpha': alpha},
            n_runs=1,
            save_folder='baseline_polynomial_fixed_topology',
            experiment_name='baseline_polynomial_fixed_topology',
            tee=True,
            seed=145767,
            formulation_arguments={
                'use_bind_first_steiner': False,
                'use_convex_hull': False,
                'use_obj_lb': False,
                'use_better_obj': False
            }
        )

        res = exp.run(multithreaded=False)
