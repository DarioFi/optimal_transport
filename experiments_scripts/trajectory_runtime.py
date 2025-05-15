import io
from contextlib import redirect_stdout

from opt_trans.experiment_manager import ExperimentManager
from opt_trans.problems.instance_generators import random_points_unit_square_with_masses
from opt_trans.formulations.relaxed_formulations import dbtq

import matplotlib.pyplot as plt

nm = ExperimentManager()

N_THREADS = 1
MULTITHREADED = False

n = 7

nm.fixed_params['instance_generator'] = random_points_unit_square_with_masses
nm.grid_params['instance_arguments'] = [{'n': n}]
# nm.grid_params['instance_arguments'] = [{'n': 6}, {'n': 7},]

nm.baron_solver(300)

nm.fixed_params['solver_options'] += " PrTimeFreq=5"

nm.fixed_params['n_runs'] = 1
nm.fixed_params['save_folder'] = "thrash/"  # Do not save
nm.fixed_params['seed'] = 53267
nm.fixed_params['tee'] = True

nm.fixed_params['experiment_name'] = 'trajectory_runtime_dbtq'
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

# Capture stdout while still printing it (tee)
results_stream = io.StringIO()

with redirect_stdout(results_stream):
    nm.run_save(MULTITHREADED, N_THREADS, exp_tee=False)

results = results_stream.getvalue()

# write to file

with open(f"result_string_{n=}.txt", "w") as f:
    f.write(results)
