from formulations.relaxed_formulations import dbt_relaxed_alpha0
from formulations.dbt import dbt_alpha_0
from experiment import Experiment
from problems.closest_counterexample import random_points_unit_square_with_masses

# todo: check baron logs for cuts

n_runs = 20
# solver_opt = "maxiter=1 maxtime=300 LBTTDo=0 "
# solver_opt = "maxiter=1 maxtime=300 TDo=0 MDo=0 LBTTDo=0 OBTTDO=0 PDo=0"
solver_opt = "maxtime=300"

exp_relaxed = Experiment(
    formulation=dbt_relaxed_alpha0,
    solver="baron",
    solver_options=solver_opt,
    instance_generator=random_points_unit_square_with_masses,
    instance_arguments={'n': 15, 'alpha': 0},
    n_runs=n_runs,
    save_folder='temp_formulation_test',
    experiment_name='test_formulation',
    tee=True,
    seed=145767,
    formulation_arguments={
        'relax_y': True,
        'relax_w': True
    }
)


exp = Experiment(
    formulation=dbt_alpha_0,
    solver="baron",
    solver_options=solver_opt,
    instance_generator=random_points_unit_square_with_masses,
    instance_arguments={'n': 5, 'alpha': 0},
    n_runs=n_runs,
    save_folder='temp_formulation_test',
    experiment_name='test_formulation',
    tee=True,
    seed=145767,
    formulation_arguments={
        'use_bind_first_steiner': False,
        'use_convex_hull': False,
        'use_obj_lb': False,
        'use_better_obj': False
    }
)


res_relax = exp_relaxed.run(True, 6)

# res = exp.run(True, 6)

res = [{'results': {'lower_bound': 0.0, 'upper_bound': 0.0}}] * n_runs

for e1, e2 in zip(res_relax, res):
    print(e1["results"]["lower_bound"], e1["results"]["upper_bound"])
    print(e2["results"]["lower_bound"], e2["results"]["upper_bound"])
    print("-------------------")

print(sum(e["results"]["lower_bound"] for e in res_relax) / n_runs)
# print number of iterations
print("Iterations:  ", sum(int(e["results"]["iterations"]) for e in res_relax) / n_runs)