from experiment import Experiment
from formulations.dbt import dbt_alpha_0, dbt
from problems.closest_counterexample import random_points_unit_square_with_masses

n_runs = 20
n = 5
exp = Experiment(
    instance_generator=random_points_unit_square_with_masses,
    instance_arguments={'n': n, "alpha": 0},
    solver='baron',
    solver_options='maxtime=300 EpsR=0.0001',
    formulation=dbt_alpha_0,
    formulation_arguments={
        'use_bind_first_steiner': False,
        'use_better_obj': False,
        'use_obj_lb': False,
        'use_convex_hull': False,
    },
    seed=23324,
    save_folder='garbage',
    experiment_name=f'dbt',
    tee=True,
    n_runs=n_runs
)

results_deg = exp.run(multithreaded=True, n_threads=4)
exp.save_to_disk(results_deg)

exp = Experiment(
    instance_generator=random_points_unit_square_with_masses,
    instance_arguments={'n': n, "alpha": 0},
    solver='baron',
    solver_options='maxtime=300 EpsR=0.0001',
    formulation=dbt_alpha_0,
    formulation_arguments={
        'use_bind_first_steiner': False,
        'use_better_obj': True,
        'use_obj_lb': False,
        'use_convex_hull': False,
    },
    seed=23324,
    save_folder='garbage',
    experiment_name=f'dbt',
    tee=True,
    n_runs=n_runs
)

results_proper = exp.run(multithreaded=True, n_threads=4)
exp.save_to_disk(results_proper)

for d, p in zip(results_deg, results_proper):
    print("obj:     ", d["results"]["objective"], p["results"]["objective"])
    print("time:    ", d["results"]["time"], p["results"]["time"])
    if not abs(d["results"]["objective"] - p["results"]["objective"]) < 1e-3:
        pass
        # todo: explore this

print(sum(d["results"]["time"] for d in results_deg) / 20)
print(sum(p["results"]["time"] for p in results_proper) / 20)

# todo : add tolerance
# todo: this looks slower like wtf?
# todo: open graphs as they look weird and the objectives are different
# (in particular check whether the obj function works)
