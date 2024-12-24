import time

from experiment import Experiment
from problems.closest_counterexample import random_points_unit_square_with_masses

from formulations.dbt import dbt_alpha_0, dbt

n = 8

n_runs = 6

solver = "baron"

solver_options = "MaxIter=1 DoLocal=1 NumLoc=-2 TDo=1 MDo=1 LBTTDo=1 OBTTDo=1 PDo=-2"
# solver_options = "DoLocal=1 TDo=1 MDo=1 LBTTDo=1 OBTTDo=1 PDo=-2"
# solver_options = "MaxIter=1 DoLocal=0 TDo=0 MDo=0 LBTTDo=1 OBTTDo=0 PDo=0"

exp = Experiment(
    instance_generator=random_points_unit_square_with_masses,
    instance_arguments={'n': n, "alpha": 0},
    solver=solver,
    solver_options=solver_options,
    formulation=dbt_alpha_0,
    formulation_arguments={
        'use_bind_first_steiner': True,
        'use_better_obj': False,
        'use_obj_lb': False,
        'use_convex_hull': False,
        'use_gurobi': False,
    },
    seed=145767,
    save_folder='gurobi_test_thrash',
    experiment_name=f'gurobi_test',
    tee=True,
    n_runs=n_runs
)

t = time.time()

n_threads = 6

results_proper = exp.run(multithreaded=True, n_threads=n_threads)
exp.save_to_disk(results_proper)

lb = [r["results"]["lower_bound"] for r in results_proper]
ub = [r["results"]["upper_bound"] for r in results_proper]

times = [r["results"]["time"] for r in results_proper]

avg_lb = sum(lb) / len(lb)
avg_ub = sum(ub) / len(ub)

print(avg_lb, avg_ub)

print(f"Elapsed time: {sum(times) / len(times):2}")
