import sys

from experiment_manager import ExperimentManager
from problems.closest_counterexample import random_points_unit_square_with_masses
from formulations.dbt import dbt, dbt_alpha_0

import itertools as it

# get alpha from cli named argument
alpha = float(sys.argv[1])

nm = ExperimentManager()

nm.fixed_params['instance_generator'] = random_points_unit_square_with_masses
nm.grid_params['instance_arguments'] = [{'n': 4, 'alpha': alpha}, {'n': 5, 'alpha':alpha}, {'n': 6, 'alpha':alpha}, {'n': 7, 'alpha':alpha}]

nm.baron_solver(300)

nm.fixed_params['n_runs'] = 50
nm.fixed_params['save_folder'] = 'runs/'

if alpha != 0:
    nm.fixed_params['formulation'] = dbt
else:
    nm.fixed_params['formulation'] = dbt_alpha_0

bools_grid = it.product([True, False], repeat=3)

nm.grid_params['formulation_arguments'] = []

for grid in bools_grid:
    nm.grid_params['formulation_arguments'].append({
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
