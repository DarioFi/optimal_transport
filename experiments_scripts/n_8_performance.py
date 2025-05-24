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
nm.grid_params['instance_arguments'] = [{'n': 8}]

nm.baron_solver(600)

nm.fixed_params['n_runs'] = 20
nm.fixed_params['save_folder'] = '../run_logs/n_8_performance/'

nm.fixed_params['seed'] = 53267
nm.fixed_params['tee'] = True

nm.fixed_params['experiment_name'] = 'n_8_performance_dbtq'
nm.fixed_params['formulation'] = dbtq

nm.fixed_params['formulation_arguments'] = {
    'relax_y': False,
    'relax_w': False,
    'disjunctive_w': False,
    'use_geometric_cut_50': True,
    'angles_constraint': False,
    'starting_position': None
}

nm.build_experiments()
nm.run_save(MULTITHREADED, N_THREADS, exp_tee=True)

nm.fixed_params['formulation_arguments'] = {
    'relax_y': False,
    'relax_w': False,
    'disjunctive_w': 1,
    'use_geometric_cut_50': True,
    'angles_constraint': False,
    'starting_position': None
}
nm.fixed_params['experiment_name'] = 'n_8_performance_dbtq_w_relax_1'

nm.build_experiments()
nm.run_save(MULTITHREADED, N_THREADS, exp_tee=True)
