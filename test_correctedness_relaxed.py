from formulations.relaxed_formulations import dbt_relaxed_alpha0
from formulations.dbt import dbt_alpha_0
from experiment import Experiment
from problems.closest_counterexample import random_points_unit_square_with_masses

# todo: check baron logs for cuts

n_runs = 40
n_threads = 6
# solver_opt = "maxiter=1 maxtime=300 LBTTDo=0 "
# solver_opt = "maxiter=1 maxtime=300 TDo=0 MDo=0 LBTTDo=0 OBTTDO=0 PDo=0"


for n in [4, 5, 6]:
    print("Running tests for n=", n)

    print("Running full solutions")
    exp = Experiment(
        formulation=dbt_relaxed_alpha0,
        solver="baron",
        solver_options="maxtime=300",
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
        }
    )

    data = exp.run(multithreaded=True, n_threads=n_threads)
    exp.save_to_disk(data)

    print("Running pre-processing only solutions")
    exp = Experiment(
        formulation=dbt_relaxed_alpha0,
        solver="baron",
        solver_options="maxtime=300 MaxIter=1",
        instance_generator=random_points_unit_square_with_masses,
        instance_arguments={'n': n, 'alpha': 0},
        n_runs=n_runs,
        save_folder='table_latex',
        experiment_name='full_solutions_preprocessing_only',
        tee=False,
        seed=145767,
        formulation_arguments={
            'relax_y': False,
            'relax_w': False,
            'disjunctive_w': False,
        }
    )

    data = exp.run(multithreaded=True, n_threads=n_threads)
    exp.save_to_disk(data)

    print("Running relaxed w")
    exp = Experiment(
        formulation=dbt_relaxed_alpha0,
        solver="baron",
        solver_options="maxtime=300",
        instance_generator=random_points_unit_square_with_masses,
        instance_arguments={'n': n, 'alpha': 0},
        n_runs=n_runs,
        save_folder='table_latex',
        experiment_name='relax_w',
        tee=True,
        seed=145767,
        formulation_arguments={
            'relax_y': False,
            'relax_w': True,
            'disjunctive_w': False,
        }
    )

    data = exp.run(multithreaded=True, n_threads=n_threads)
    exp.save_to_disk(data)

    print("Running full relaxation")
    exp = Experiment(
        formulation=dbt_relaxed_alpha0,
        solver="baron",
        solver_options="maxtime=300",
        instance_generator=random_points_unit_square_with_masses,
        instance_arguments={'n': n, 'alpha': 0},
        n_runs=n_runs,
        save_folder='table_latex',
        experiment_name='relax',
        tee=False,
        seed=145767,
        formulation_arguments={
            'relax_y': True,
            'relax_w': True,
            'disjunctive_w': False,
        }
    )

    data = exp.run(multithreaded=True, n_threads=n_threads)
    exp.save_to_disk(data)
