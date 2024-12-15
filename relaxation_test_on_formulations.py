from formulations.dbt import dbt_alpha_0
from experiment_manager import ExperimentManager, Experiment
from problems.closest_counterexample import random_points_unit_square_with_masses
import itertools as it

baron_example_options = "DoLocal=0 TDo=0 MDo=0 LBTTDo=0 OBTTDo=0 PDo=0 MaxIter=1"

# todo:


s0 = [
    "MaxIter=1"
]

s1 = [
    "DoLocal=1 NumLoc=-2",
    "DoLocal=1 NumLoc=50",
    "DoLocal=0 NumLoc=-2",
]

s2 = [
    "TDo=0 MDo=0 LBTTDo=0 OBTTDo=0 PDo=0",
    "TDo=1 MDo=1 LBTTDo=1 OBTTDo=1 PDo=-2",
    "TDo=1 MDo=1 LBTTDo=1 OBTTDo=1 PDo=-1",
    "TDo=1 MDo=1 LBTTDo=1 OBTTDo=1 PDo=0",
    "TDo=1 MDo=1 LBTTDo=1 OBTTDo=1 PDo=10",
]


solver_options_combos = list(map(lambda x: " ".join(x), it.product(s0, s1, s2)))

nm = ExperimentManager()

nm.fixed_params['instance_generator'] = random_points_unit_square_with_masses
nm.fixed_params['instance_arguments'] = {'n': 5, 'alpha': 0}


nm.grid_params['solver_options'] = solver_options_combos


nm.fixed_params['solver'] = 'baron'
nm.fixed_params['tee'] = False

nm.fixed_params['n_runs'] = 50
nm.fixed_params['save_folder'] = 'relaxation_runs_formulation_grid'

nm.fixed_params['seed'] = 145767
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

# nm.fixed_params['formulation_arguments'] = {
#     'use_bind_first_steiner': False,
#     'use_convex_hull': False,
#     'use_obj_lb': False,
#     'use_better_obj': False
# }

nm.grid_params['solver_options'] = solver_options_combos

nm.fixed_params['experiment_name'] = 'relaxation_test'
nm.build_experiments()

print(f"{len(nm.queued_experiments)=}")

nm.run_save(True, 8, bar=True, exp_tee=False, accumulate=20)

