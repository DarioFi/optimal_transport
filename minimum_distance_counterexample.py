import pyomo.environ as pyo

from formulations.gmmx import gmmx
from formulations.dbt import dbt, dbt_alpha_0
from formulations.relaxed_formulations import dbt_relaxed_alpha0
from experiment import Experiment, extract_results
from data_handling.experiment_data import ExperimentData
from data_handling.visualize_graph import visualize

terminals_unnormalized = [
    [0, 0],
    [0, .5],
    [0, 0.8],
    [0.1, 1],
    [0.7, 0.1],
    [0.6, 1],
    [0.8, 0.9],
    [0.5, 0.5]
]

max_t = max(max(tu) for tu in terminals_unnormalized)

terminals = [tuple(t_i / max_t for t_i in t) for t in terminals_unnormalized]

del terminals_unnormalized
# masses = [.5, .5, -.5, -.5]


masses = [-1] + [1 / (len(terminals) - 1) for _ in range(len(terminals) - 1)]

maximum_degree = 3
alpha = 0


def gen_instance():
    return {
        'terminals': terminals,
        'masses': masses,
        'alpha': alpha
    }


gmmx_arg = {
    'maximum_degree': maximum_degree,
    'use_bind_first_steiner': True,
    'use_obj_lb': False
}

dbt_alpha0_arg = {
    "relax_y": False,
    "relax_w": False,
    "disjunctive_w": False,
    "use_geometric_cut_50": True,
    'angles_constraint': False
}

# TODO: SET GEOSTEINER AND FEED INITIAL SOLUTION CORRECT


dbt_arg = {
    # use_bind_first_steiner, use_obj_lb, use_convex_hull, use_better_obj
    'use_bind_first_steiner': True,
    'use_obj_lb': False,
    'use_convex_hull': True,
    'use_better_obj': True,
    'use_gurobi': False
}

baron_options = 'epsR=0.001 threads=6'
gurobi_options = 'MIPGap=0.01 logfile="gurobi.log" timelimit=30'

solver = 'baron'
# solver = 'gurobi'

exp = Experiment(
    instance_generator=gen_instance,
    instance_arguments={},
    solver=solver,
    solver_options=baron_options if solver == 'baron' else gurobi_options,
    formulation=dbt_relaxed_alpha0,
    n_runs=1,
    formulation_arguments=dbt_alpha0_arg,
    seed=0,
    save_folder='temporary',
    experiment_name=f'minimum_distance_counterexample',
    tee=True
)

res = exp.run(multithreaded=False)

optimal_variables = res[0]['results']['variables']
res_data = ExperimentData.from_json(res[0])


visualize(res_data)

# from geo_steiner_wrapper.geo_steiner_wrap import EST
#
# results = EST(terminals)
# print(results['optimal_cost'])

new_ops = dbt_alpha0_arg
new_ops['starting_position'] = optimal_variables

exp = Experiment(
    instance_generator=gen_instance,
    instance_arguments={},
    solver=solver,
    solver_options=baron_options if solver == 'baron' else gurobi_options,
    formulation=dbt_relaxed_alpha0,
    n_runs=1,
    formulation_arguments=new_ops,
    seed=0,
    save_folder='temporary',
    experiment_name=f'minimum_distance_counterexample',
    tee=True
)

res = exp.run(multithreaded=False)


