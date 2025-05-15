from opt_trans.problems.closest_counterexample import random_points_unit_square_with_masses
from opt_trans.experiment import Experiment
from opt_trans.formulations.dbt import dbt
import time

timelimit = 10

exp = Experiment(random_points_unit_square_with_masses, {"n": 6}, 'baron', f'maxtime={timelimit}', dbt,
                 formulation_arguments={
                     'alpha': 0.5, 'maximum_degree': 3,
                     'use_bind_first_steiner': False, 'use_obj_lb': False, 'use_convex_hull': False},
                 seed=0, save_folder=None, experiment_name="Converging Steiner", tee=True, n_runs=2
                 )

res = exp.run(multithreaded=True, n_threads=2)

assert len(res) == 2

print(f"{time.process_time()=}")
for i, r in enumerate(res):
    print(f"Experiment {i=}")
    print("WC time: ", r["results"]["wallclock_time"])
    print("Real time: ", r["results"]["time"])

t = [r["results"]["wallclock_time"] for r in res]

assert all(abs(t_i - timelimit) < 1 for t_i in t)
