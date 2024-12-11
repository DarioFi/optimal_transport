# idea: disable pre processing by setting max 1 iteration in the solver
import math

from formulations.dbt import dbt
from experiment import Experiment
from problems.closest_counterexample import random_points_unit_square_with_masses

# todo: approaches are "MaxIter = 1" or "FirstFeas=1" and "NumFeas=1"
# todo: branching priorities to set

# solver_opt = "MaxIter=1"
# solver_opt = "FirstFeas=1 NumFeas=1"
solver_opt = "MaxTime=60"

# "LPSOL=gurobi"

n = 5
n_runs = 1

exp = Experiment(
    instance_generator=random_points_unit_square_with_masses,
    instance_arguments={'n': n, "alpha": 0.5},
    solver='baron',
    solver_options=solver_opt,
    formulation=dbt,
    formulation_arguments={
        'use_bind_first_steiner': True,
        'use_better_obj': True,
        'use_obj_lb': False,
        'use_convex_hull': True,
    },
    seed=23324,
    save_folder='gurobi_test',
    experiment_name=f'relaxation_test',
    tee=True,
    n_runs=n_runs
)

results_proper = exp.run(multithreaded=False, n_threads=4)

for x in results_proper:
    try:
        gap = x['results']['upper_bound'] / x['results']['lower_bound']
    except ZeroDivisionError:
        gap = math.inf

    print(f"{x['results']['lower_bound']:4f} {x['results']['upper_bound']:4f} Gap: {gap:4f} Time: {x['results']['time']:2f}")


# todo: question, why does bind first steiner worsen the results?