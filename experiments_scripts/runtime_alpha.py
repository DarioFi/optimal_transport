import numpy as np

from opt_trans.experiment_manager import ExperimentManager
from opt_trans.problems.instance_generators import random_points_unit_square_with_masses
from opt_trans.formulations.dbt import dbt_alpha_0
from opt_trans.formulations.relaxed_formulations import dbtq, dbtq_with_flows

nm = ExperimentManager()

N_THREADS = 6
MULTITHREADED = True

nm.fixed_params['instance_generator'] = random_points_unit_square_with_masses

nm.grid_params['instance_arguments'] = [{'n': 6, 'alpha': a} for a in np.linspace(0.1, 0.95, 18)]

nm.baron_solver(300)

nm.fixed_params['n_runs'] = 20
nm.fixed_params['save_folder'] = '../run_logs/runtime_alpha/'

nm.fixed_params['seed'] = 53267
nm.fixed_params['tee'] = True

nm.fixed_params['experiment_name'] = 'runtime_alpha_dbt'
nm.fixed_params['formulation'] = dbtq_with_flows

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
