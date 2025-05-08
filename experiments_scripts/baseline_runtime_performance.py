# we weant to compare the runtime performance of different baseline formulations for multiple n
# this test is done at alpha = 0

from opt_trans.experiment_manager import ExperimentManager
from opt_trans.problems.instance_generators import random_points_unit_square_with_masses
from opt_trans.formulations.dbt import dbt_alpha_0
from opt_trans.formulations.relaxed_formulations import dbtq

nm = ExperimentManager()

N_THREADS = 6
MULTITHREADED = True

nm.fixed_params['instance_generator'] = random_points_unit_square_with_masses
nm.grid_params['instance_arguments'] = [{'n': 4}, {'n': 5}, {'n': 6}, {'n': 7}]

nm.baron_solver(300)

nm.fixed_params['n_runs'] = 20
nm.fixed_params['save_folder'] = '../run_logs/baseline_runtime_performance/'

nm.fixed_params['seed'] = 53267
nm.fixed_params['tee'] = True

# add other too

nm.fixed_params['experiment_name'] = 'baseline_runtime_performance_dbt'
nm.fixed_params['formulation'] = dbt_alpha_0
nm.fixed_params['formulation_arguments'] = {
    'use_bind_first_steiner': False,
    'use_convex_hull': False,
    'use_obj_lb': False,
    'use_better_obj': True
}

nm.build_experiments()
nm.run_save(MULTITHREADED, N_THREADS, exp_tee=True)

nm.fixed_params['experiment_name'] = 'baseline_runtime_performance_dbtq'
nm.fixed_params['formulation'] = dbtq

nm.fixed_params['formulation_arguments'] = {
    'relax_y': False,
    'relax_w': False,
    'disjunctive_w': False,
    'use_geometric_cut_50': False,
    'angles_constraint': False,
    'starting_position': None
}

nm.build_experiments()
nm.run_save(MULTITHREADED, N_THREADS, exp_tee=True)
