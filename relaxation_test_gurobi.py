from formulations.dbt import dbt_alpha_0
from experiment_manager import ExperimentManager, Experiment
from problems.closest_counterexample import random_points_unit_square_with_masses
import itertools as it

# todo: add maxiteration

all_combos = {
    'DoLocal': [0, 1],
    'PDo': [-2, 0, -1, 5, 10, 20],
}

keys, values = zip(*all_combos.items())
combinations = list(it.product(*values))

print(f"{len(combinations)=}")

# Combine each combination into a string
formatted_combinations = [
    "NodeLimit=1 " + " ".join(f"{key}={value}" for key, value in zip(keys, combo)) for combo in combinations
]

nm = ExperimentManager()

nm.fixed_params['instance_generator'] = random_points_unit_square_with_masses
nm.fixed_params['instance_arguments'] = {'n': 5, 'alpha': 0}

nm.fixed_params['solver'] = 'gurobi_persistent'
nm.fixed_params['tee'] = True

nm.fixed_params['n_runs'] = 50
nm.fixed_params['save_folder'] = 'relaxation_runs'

nm.fixed_params['seed'] = 145767
nm.fixed_params['formulation'] = dbt_alpha_0

# bools_grid = it.product([True, False], repeat=3)
#
# nm.grid_params['formulation_arguments'] = []
# for grid in bools_grid:
#     nm.grid_params['formulation_arguments'].append({
#         'use_bind_first_steiner': grid[0],
#         'use_convex_hull': grid[1],
#         'use_obj_lb': False,
#         'use_better_obj': grid[2]
#     })

nm.fixed_params['formulation_arguments'] = {
    'use_bind_first_steiner': False,
    'use_convex_hull': False,
    'use_obj_lb': False,
    'use_better_obj': False,
    'use_gurobi': True
}

nm.grid_params['solver_options'] = formatted_combinations

nm.fixed_params['experiment_name'] = 'relaxation_test'
nm.build_experiments()

print(f"{len(nm.queued_experiments)=}")

# nm.run_save(True, 8, bar=True, exp_tee=True, accumulate=20)


exp = Experiment(
    formulation=dbt_alpha_0,
    solver="gurobi_persistent",
    solver_options={
        "NodeLimit": 1
    },
    instance_generator=random_points_unit_square_with_masses,
    instance_arguments={'n': 4, 'alpha': 0},
    n_runs=1,
    save_folder='garbage',
    experiment_name='relaxation',
    tee=True,
    seed=145767,
    formulation_arguments={
        'use_bind_first_steiner': False,
        'use_convex_hull': False,
        'use_obj_lb': False,
        'use_better_obj': False,
        'use_gurobi': True
    }
)

results = exp.run(False, n_threads=8)
exp.save_to_disk(results)
