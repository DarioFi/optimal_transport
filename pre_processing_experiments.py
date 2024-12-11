# idea: disable pre processing by setting max 1 iteration in the solver
import sys
import itertools as it

from experiment_manager import ExperimentManager
from formulations.dbt import dbt, dbt_alpha_0
from experiment import Experiment
from problems.closest_counterexample import random_points_unit_square_with_masses

# todo: approaches are "MaxIter = 1" or "FirstFeas=1" and "NumFeas=1"
# todo: branching priorities to set

# solver_opt = "MaxIter=1"
solver_opt = "FirstFeas=1 NumFeas=1"
# solver_opt = "MaxTime=60"
# alpha = float(sys.argv[1])

nm = ExperimentManager()

nm.fixed_params['instance_generator'] = random_points_unit_square_with_masses

n_alpha = it.product([4, 5, 6, 7], [0, 0.2, 0.5, 0.8])
nm.grid_params['instance_arguments'] = []

for n, alpha in n_alpha:
    nm.grid_params['instance_arguments'].append({'n': n, 'alpha': alpha})

nm.fixed_params['solver'] = 'baron'
nm.fixed_params['solver_options'] = solver_opt
nm.fixed_params['tee'] = False


nm.fixed_params['n_runs'] = 50
nm.fixed_params['save_folder'] = 'pre_processing_runs'

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

nm.fixed_params['seed'] = 145767
nm.fixed_params['experiment_name'] = 'relaxation_test'


nm.build_experiments()
nm.run_save(True, 8)

print(f"{len(nm.queued_experiments)=}")
# todo: question, why does bind first steiner worsen the results?


if False:
    exp = Experiment(
        instance_generator=random_points_unit_square_with_masses,
        instance_arguments={'n': 5, 'alpha': 0.2},
        formulation=dbt,
        formulation_arguments={
            'use_bind_first_steiner': True,
            'use_convex_hull': True,
            'use_obj_lb': False,
            'use_better_obj': False
        },
        solver='baron',
        solver_options=solver_opt,
        tee=False,
        seed=145767,
        n_runs=50,
        save_folder='garbage',
        experiment_name='garbage'
    )

    res = exp.run(True, 8)


    exp.save_to_disk(res)