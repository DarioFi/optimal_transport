# we weant to compare the runtime performance of different baseline formulations for multiple n
# this test is done at alpha = 0

from opt_trans.experiment_manager import ExperimentManager
from opt_trans.problems.instance_generators import random_points_unit_square_with_masses
from opt_trans.formulations.dbt import dbt_alpha_0
from opt_trans.formulations.relaxed_formulations import dbtq, dbtq_with_flows

nm = ExperimentManager()

N_THREADS = 6
MULTITHREADED = True

nm.fixed_params['instance_generator'] = random_points_unit_square_with_masses
nm.grid_params['instance_arguments'] = [{'n': 4}, {'n': 5}, {'n': 6}, {'n': 7}]

nm.baron_solver(300)

nm.fixed_params['n_runs'] = 20
nm.fixed_params['save_folder'] = '../run_logs/baseline_runtime_performance_logarithm/'

nm.fixed_params['seed'] = 53267
nm.fixed_params['tee'] = True


nm.fixed_params['experiment_name'] = 'baseline_runtime_performance_dbtq_flow_logarithm'
nm.fixed_params['formulation'] = dbtq_with_flows

nm.grid_params['formulation_arguments'] = []

base_arg = {
    'relax_y': False,
    'relax_w': False,
    'disjunctive_w': False,
    'use_geometric_cut_50': False,
    'angles_constraint': False,
    'starting_position': None,
    'use_log_obj': True
}
# for log_mul in [100]:
for log_mul in [10, 20, 40, 100]:
    nm.grid_params['formulation_arguments'].append(
        {**base_arg, 'log_multiplier': log_mul}
    )

nm.build_experiments()
nm.run_save(MULTITHREADED, N_THREADS, exp_tee=True)

