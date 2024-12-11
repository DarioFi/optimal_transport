from experiment import Experiment
from problems.closest_counterexample import random_points_unit_square_with_masses

from formulations.dbt import dbt_alpha_0, dbt

n = 3

n_runs = 1

solvers = [
    "gurobi_persistent",
    "baron",
]

solver_options = [
    {"threads": 8},
    "",
]

for solver, opts in zip(solvers, solver_options):
    exp = Experiment(
        instance_generator=random_points_unit_square_with_masses,
        instance_arguments={'n': n, "alpha": 0},
        solver=solver,
        solver_options=opts,
        formulation=dbt_alpha_0,
        formulation_arguments={
            'use_bind_first_steiner': False,
            'use_better_obj': False,
            'use_obj_lb': False,
            'use_convex_hull': False,
        },
        seed=23324,
        save_folder='gurobi_test_thrash',
        experiment_name=f'gurobi_test',
        tee=True,
        n_runs=n_runs
    )

    results_proper = exp.run(multithreaded=False, n_threads=4)
    exp.save_to_disk(results_proper)
