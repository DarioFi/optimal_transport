from experiment_manager import ExperimentManager
from problems.closest_counterexample import random_points_unit_square_with_masses
from formulations.dbt import dbt

nm = ExperimentManager()

nm.fixed_params['instance_generator'] = random_points_unit_square_with_masses
nm.grid_params['instance_arguments'] = [{'n': 6}, {'n': 7}]

nm.baron_solver(300)

nm.fixed_params['n_runs'] = 50
nm.fixed_params['save_folder'] = 'runs'

nm.fixed_params['formulation'] = dbt
nm.grid_params['formulation_arguments'] = [
    {
        'maximum_degree': 3,
        'alpha': .5,
        'use_bind_first_steiner': True,
        'use_convex_hull': True,
        'use_obj_lb': False
    },
    {
        'maximum_degree': 3,
        'alpha': .5,
        'use_bind_first_steiner': True,
        'use_convex_hull': False,
        'use_obj_lb': False
    },
    {
        'maximum_degree': 3,
        'alpha': .5,
        'use_bind_first_steiner': False,
        'use_convex_hull': True,
        'use_obj_lb': False
    },
    {
        'maximum_degree': 3,
        'alpha': .5,
        'use_bind_first_steiner': False,
        'use_convex_hull': False,
        'use_obj_lb': False
    },

]

nm.random_seed()
nm.fixed_params['tee'] = False
nm.fixed_params['experiment_name'] = 'dbt_baseline_test'

nm.fixed_params['solver'] = 'baron'

nm.build_experiments()

nm.run_save(True, 8)
