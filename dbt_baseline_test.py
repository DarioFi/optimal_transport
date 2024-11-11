from experiment_manager import ExperimentManager
from problems.closest_counterexample import random_points_unit_square_with_masses
from formulations.dbt import dbt

import itertools as it

nm = ExperimentManager()

nm.fixed_params['instance_generator'] = random_points_unit_square_with_masses
nm.grid_params['instance_arguments'] = [{'n': 4}, {'n': 5}, {'n': 6}, {'n': 7}]

nm.baron_solver(300)

nm.fixed_params['n_runs'] = 50
nm.fixed_params['save_folder'] = 'runs'

nm.fixed_params['formulation'] = dbt

bools_grid = it.product([True, False], repeat=3)

nm.grid_params['formulation_arguments'] = []

for grid in bools_grid:
    nm.grid_params['formulation_arguments'].append({
        'alpha': .5,
        'use_bind_first_steiner': grid[0],
        'use_convex_hull': grid[1],
        'use_obj_lb': False,
        'use_better_obj': grid[2]
    })

nm.fixed_params['seed'] = 53267
nm.fixed_params['tee'] = False
nm.fixed_params['experiment_name'] = 'dbt_baseline_test'

nm.build_experiments()

nm.run_save(True, 12)
