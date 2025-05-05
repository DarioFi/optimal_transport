# we aim to find a sequence of converging steiner points, fixing alpha
from pyomo.opt import SolverFactory

from opt_trans.data_handling.experiment_data import ExperimentData
from opt_trans.formulations.dbt import dbt
from opt_trans.experiment import Experiment
from opt_trans.data_handling.visualize_graph import visualize


def gen_instance(height):
    inv = 1 / height
    terminals = [(.5, .5), (0, .5 + inv), (0, .5 - inv), (1, .5 - inv), (1, .5 + inv)]
    masses = [-1, .25, .25, .25, .25]

    maximum_degree = 3
    alpha = .5

    return {
        'terminals': terminals,
        'masses': masses,
        # 'maximum_degree': maximum_degree,
        'alpha': alpha
    }


def converging_steiner():
    data = []
    for height_ratio in [4, 3, 2.5, 2.2, 2.1, 2.05, 2.01, 2]:
        exp = Experiment(gen_instance, {"height": height_ratio}, 'baron', '', dbt,
                     formulation_arguments={'use_bind_first_steiner': False, 'use_obj_lb': False, 'use_convex_hull': False,
                                            'use_better_obj': True},
                     seed=0, save_folder=None, experiment_name="Converging Steiner", tee=True, n_runs=1
                         )

        results = exp.run(multithreaded=False)

        data.append(ExperimentData.from_json(results[0]))

        print(f"Height ratio: {height_ratio}")
        visualize(data[-1])
        exit()



if __name__ == '__main__':
    converging_steiner()
