from formulations.relaxed_formulations import dbt_relaxed_alpha0
from formulations.dbt import dbt_alpha_0
from experiment import Experiment
from problems.closest_counterexample import random_points_unit_square_with_masses

n_runs = 10
n_threads = 6
n = 5
# solver_opt = "maxiter=1 maxtime=300 LBTTDo=0 "
# solver_opt = "maxiter=1 maxtime=300 TDo=0 MDo=0 LBTTDo=0 OBTTDO=0 PDo=0"

solver_opt = "maxtime=60"

print("Running full solutions no cut")
exp = Experiment(
    formulation=dbt_relaxed_alpha0,
    solver="baron",
    solver_options=solver_opt,
    instance_generator=random_points_unit_square_with_masses,
    instance_arguments={'n': n, 'alpha': 0},
    n_runs=n_runs,
    save_folder='table_latex',
    experiment_name='full_solutions',
    tee=True,
    seed=145767,
    formulation_arguments={
        'relax_y': False,
        'relax_w': False,
        'disjunctive_w': False,
        'use_geometric_cut_50': False
    }
)

data_no_cut = exp.run(multithreaded=True, n_threads=n_threads)


print("Running full solutions cut")
exp = Experiment(
    formulation=dbt_relaxed_alpha0,
    solver="baron",
    solver_options=solver_opt,
    instance_generator=random_points_unit_square_with_masses,
    instance_arguments={'n': n, 'alpha': 0},
    n_runs=n_runs,
    save_folder='table_latex',
    experiment_name='full_solutions',
    tee=True,
    seed=145767,
    formulation_arguments={
        'relax_y': False,
        'relax_w': False,
        'disjunctive_w': False,
        'use_geometric_cut_50': True
    }
)

data_cut = exp.run(multithreaded=True, n_threads=n_threads)


# assert that the solutions are the same

for i in range(n_runs):
    # print termination condition
    t_no_cut = data_no_cut[i]['results']['termination_condition']
    t_cut = data_cut[i]['results']['termination_condition']
    if t_no_cut != t_cut:
        print(f"{t_no_cut} != {t_cut}")
        print(f"{data_no_cut[i]['results']['lower_bound']} == {data_cut[i]['results']['lower_bound']}")
        print(f"{data_no_cut[i]['results']['upper_bound']} == {data_cut[i]['results']['upper_bound']}")
    else:
        print(f"{data_no_cut[i]['results']['lower_bound']} == {data_cut[i]['results']['lower_bound']}")
        assert abs(data_no_cut[i]['results']['lower_bound'] - data_cut[i]['results']['lower_bound']) < 1e-5
        assert abs(data_no_cut[i]['results']['upper_bound'] - data_cut[i]['results']['upper_bound']) < 1e-5


print(f"Average time no cut: {sum([x['results']['time'] for x in data_no_cut]) / n_runs}")
print(f"Average time cut: {sum([x['results']['time'] for x in data_cut]) / n_runs}")