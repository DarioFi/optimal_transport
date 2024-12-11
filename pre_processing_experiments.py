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

a = {'status': 'ok', 'termination_condition': 'optimal', 'objective': 1.9077141554862098, 'lower_bound': 0.0,
     'upper_bound': 1.90771415548621, 'iterations': '-1', 'time': 0.01, 'wallclock_time': 0.01, 'variables': {
        'x': {'(5, 0)': 0.07794633541586027, '(5, 1)': 0.2986617272310479, '(6, 0)': 0.07794633541586027,
              '(6, 1)': 0.2986617272310479, '(7, 0)': 0.07794633541586027, '(7, 1)': 0.2986617272310479},
        'y': {'(0, 5)': 1.0, '(0, 6)': 0.0, '(0, 7)': 0.0, '(1, 5)': 1.0, '(1, 6)': 0.0, '(1, 7)': 0.0, '(2, 5)': 0.0,
              '(2, 6)': 0.0, '(2, 7)': 1.0, '(3, 5)': 0.0, '(3, 6)': 0.0, '(3, 7)': 1.0, '(4, 5)': 0.0, '(4, 6)': 1.0,
              '(4, 7)': 0.0, '(5, 6)': 1.0, '(5, 7)': 0.0, '(6, 7)': 1.0},
        'f': {'(0, 5)': 1.0, '(0, 6)': 0.0, '(0, 7)': 0.0, '(1, 5)': 0.4586707729471681, '(1, 6)': 0.0, '(1, 7)': 0.0,
              '(2, 5)': 0.0, '(2, 6)': 0.0, '(2, 7)': 0.1462320903447258, '(3, 5)': 0.0, '(3, 6)': 0.0,
              '(3, 7)': 0.10021986097945254, '(4, 5)': 0.0, '(4, 6)': 0.29487727572865347, '(4, 7)': 0.0,
              '(5, 6)': 0.5413292270528318, '(5, 7)': 0.0, '(6, 7)': 0.24645195132417835},
        'c': {'(5, 0)': 0.0, '(5, 1)': 0.0, '(5, 2)': 1.0, '(5, 3)': 0.0, '(5, 4)': 0.0, '(6, 0)': 0.0, '(6, 1)': 0.0,
              '(6, 2)': 1.0, '(6, 3)': 0.0, '(6, 4)': 0.0, '(7, 0)': 0.0, '(7, 1)': 0.0, '(7, 2)': 1.0, '(7, 3)': 0.0,
              '(7, 4)': 0.0}}}
