from experiment import Experiment
from formulations.dbt import dbt
from problems.closest_counterexample import random_points_unit_square_with_masses

n = 5
n_runs = 20

exp = Experiment(
    instance_generator=random_points_unit_square_with_masses,
    instance_arguments={'n': n, "alpha": 0.5},
    solver='gurobi_persistent',
    solver_options='TimeLimit=300',
    formulation=dbt,
    formulation_arguments={
        'use_bind_first_steiner': False,
        'use_better_obj': True,
        'use_obj_lb': False,
        'use_convex_hull': False,
    },
    seed=23324,
    save_folder='gurobi_test',
    experiment_name=f'gurobi_test',
    tee=True,
    n_runs=n_runs
)

results_proper = exp.run(multithreaded=False, n_threads=4)
exp.save_to_disk(results_proper)
