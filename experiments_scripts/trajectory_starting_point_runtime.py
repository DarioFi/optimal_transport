import io
import os

from opt_trans.formulations.relaxed_formulations import dbtq
from opt_trans.experiment import Experiment, extract_results
from opt_trans.data_handling.experiment_data import ExperimentData
from opt_trans.data_handling.visualize_graph import visualize
from opt_trans.problems.instance_generators import random_points_unit_square_with_masses

terminals_unnormalized = [
    [0, 0],
    [0, .5],
    [0, 0.8],
    [0.1, 1],
    [0.7, 0.1],
    [0.6, 1],
    [0.8, 0.9],
]

max_t = max(max(tu) for tu in terminals_unnormalized)

terminals = [tuple(t_i / max_t for t_i in t) for t in terminals_unnormalized]

del terminals_unnormalized

masses = [-1] + [1 / (len(terminals) - 1) for _ in range(len(terminals) - 1)]

maximum_degree = 3
alpha = 0

import random
random.seed(1319)

for iter in range(9):
    instance_data = random_points_unit_square_with_masses(n=7)


    def gen_instance():
        return instance_data


    dbtq_arg = {
        "relax_y": False,
        "relax_w": False,
        "disjunctive_w": False,
        "use_geometric_cut_50": True,
        'angles_constraint': False
    }

    baron_options = 'epsR=0.001 threads=6 PrTimeFreq=2'
    gurobi_options = 'MIPGap=0.01 logfile="gurobi.log" timelimit=30'

    solver = 'baron'

    exp = Experiment(
        instance_generator=gen_instance,
        instance_arguments={},
        solver=solver,
        solver_options=baron_options if solver == 'baron' else gurobi_options,
        formulation=dbtq,
        n_runs=1,
        formulation_arguments=dbtq_arg,
        seed=0,
        save_folder='temporary',
        experiment_name=f'minimum_distance_counterexample',
        tee=True
    )

    results_stream = io.StringIO()
    from contextlib import redirect_stdout

    with redirect_stdout(results_stream):
        res = exp.run(multithreaded=False)
    results = results_stream.getvalue()

    with open(f"experiments_scripts/starting_point_strings/no_starting_point_experiment_{iter=}.txt", "w") as f:
        f.write(results)

    optimal_variables = res[0]['results']['variables']
    res_data = ExperimentData.from_json(res[0])

    visualize(res_data)

    new_ops = dbtq_arg
    new_ops['starting_position'] = optimal_variables

    exp = Experiment(
        instance_generator=gen_instance,
        instance_arguments={},
        solver=solver,
        solver_options=baron_options if solver == 'baron' else gurobi_options,
        formulation=dbtq,
        n_runs=1,
        formulation_arguments=new_ops,
        seed=0,
        save_folder='temporary',
        experiment_name=f'minimum_distance_counterexample',
        tee=True
    )

    results_stream = io.StringIO()
    from contextlib import redirect_stdout

    with redirect_stdout(results_stream):
        res = exp.run(multithreaded=False)
    results = results_stream.getvalue()

    with open(f"experiments_scripts/starting_point_strings/starting_point_experiment_{iter=}.txt", "w") as f:
        f.write(results)
