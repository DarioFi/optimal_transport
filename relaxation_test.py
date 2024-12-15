from formulations.dbt import dbt_alpha_0
from experiment_manager import ExperimentManager, Experiment
from problems.closest_counterexample import random_points_unit_square_with_masses
import itertools as it

"""
Gurobi complained about the form of those constraints but it accepted them when split with auxiliary variables.

For Baron I tested with the default options and maxIter=1, but I will test the others and come back with some better answers.

We have some decent upper bounds in general, for instance by fixing the topology and just optimizing over the position of the Steiner points we get somewhat good solutions but I get some actual data to discuss this.

In short, I will prepare some actual data to support those facts, otherwise it is a bit pointless. 
Thank you for the clarifications!


     
for Gurobi. limit yourself to alpha = 0 so that the only nonlinearities you have are bilinears and second-order cone constraints ... does Gurobi complain about these? make sure to use Gurobi 12.

BARON: I am not understanding, I thought you said if you deactivate local search and range reduction, baron gives a lower bound of zero ... is that not the case? when do you get a zero lower bound?

A good starting point in general can help, because BARON  can use it to generate good upper bounds, we dont have such points though, do we? 

I still have not tested Gurobi extensively, as it has some tighter restrictions on the kind of non-linear constraints I can feed it, so I had to add some auxiliary variables and I should play with them a bit more.
For now, I tested the default options for Baron, so both local search and range reduction are active but "BARON decides the number of local searches in preprocessing based on problem and NLP solver characteristics." so it might not improve at all.
I am also doing all those tests without specifying a starting point, I am assuming it should not change anything for the lower bound but in the future would it make sense to experiment with it?


Great, that's very good, this means without other tools, baron returns a trivial lower bound of zero. Now you want to know what makes baron's bounds improve, there are two possibilities:
(i) local search
(ii) range reductions
You should turn them on one at a time to see which one is responsible for the improvement.

interesting, about Gurobi, I'd be curious to see the results, as I was expecting for this problem class Gurobi would perform better ...

Hi,

yes, Monday would work!

I am also testing after terminating at the root node but most of the time it just gives a lower bound of 0 so I thought that it could also make senso to check the first feasible solution.
I managed to get Gurobi to work, but it seems slower than baron for the case alpha=0, in the next days I will do some more testing to compare the solvers.

Best,
Dario

For testing the convex relaxation, why do you care about feasible solutions?
You should make baron as basic as possible, turn off local search and range reduction and terminate after the root node.
Are you able to test with Gurobi? did you fix the license issue?


I have some computational results and I would like to discuss them during a call, would it be possible to schedule it? I am available on Friday at the usual time and basically everyday next week.
Also, for testing just the convex relaxation I am running baron with arguments to stop whenever it finds the first feasible solution, is this a reasonable way? Usually it is not able to find a solution during pre-processing.

"""
baron_example_options = "DoLocal=0 TDo=0 MDo=0 LBTTDo=0 OBTTDo=0 PDo=0 MaxIter=1"

# todo:
# DoLocal and NumLoc
# Tree Management??
# Range reduction: TDo, MDo, LBTTDo, OBTTDo, PDo



all_combos = {
    'DoLocal': [0, 1],
    'NumLoc': [-2, 10],  # todo whatt???
    'TDo': [0, 1],
    'MDo': [0, 1],
    'LBTTDo': [0, 1],
    'OBTTDo': [0, 1],
    'PDo': [-2, 0, -1, 5, 10, 20],
}

keys, values = zip(*all_combos.items())
combinations = list(it.product(*values))



print(f"{len(combinations)=}")

# Combine each combination into a string
formatted_combinations = [
    "MaxIter=1 " + " ".join(f"{key}={value}" for key, value in zip(keys, combo)) for combo in combinations
]

nm = ExperimentManager()

nm.fixed_params['instance_generator'] = random_points_unit_square_with_masses
nm.fixed_params['instance_arguments'] = {'n': 5, 'alpha': 0}

nm.fixed_params['solver'] = 'baron'
nm.fixed_params['tee'] = True

nm.fixed_params['n_runs'] = 50
nm.fixed_params['save_folder'] = 'relaxation_runs'

nm.fixed_params['seed'] = 145767
nm.fixed_params['formulation'] = dbt_alpha_0

# bools_grid = it.product([True, False], repeat=3)
#
# nm.grid_params['formulation_arguments'] = []
# for grid in bools_grid:
#     nm.grid_params['formulation_arguments'].append({
#         'use_bind_first_steiner': grid[0],
#         'use_convex_hull': grid[1],
#         'use_obj_lb': False,
#         'use_better_obj': grid[2]
#     })

nm.fixed_params['formulation_arguments'] = {
    'use_bind_first_steiner': False,
    'use_convex_hull': False,
    'use_obj_lb': False,
    'use_better_obj': False
}

nm.grid_params['solver_options'] = formatted_combinations

nm.fixed_params['experiment_name'] = 'relaxation_test'
nm.build_experiments()

print(f"{len(nm.queued_experiments)=}")

# nm.run_save(True, 8, bar=True, exp_tee=True, accumulate=20)


exp = Experiment(
    formulation=dbt_alpha_0,
    solver="baron",
    solver_options="",
    instance_generator=random_points_unit_square_with_masses,
    instance_arguments={'n': 5, 'alpha': 0},
    n_runs=50,
    save_folder='relaxation_runs_real_solutions',
    experiment_name='relaxation',
    tee=True,
    seed=145767,
    formulation_arguments={
        'use_bind_first_steiner': False,
        'use_convex_hull': False,
        'use_obj_lb': False,
        'use_better_obj': False
    }
)

results = exp.run(True, n_threads=8)
exp.save_to_disk(results)
