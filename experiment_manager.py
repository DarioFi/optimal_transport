import copy
from typing import Dict, List, Optional
import itertools

import random

import tqdm

import formulations.dbt
import formulations.gmmx
from experiment import Experiment
from problems.closest_counterexample import random_points_unit_square_with_masses


# this class is a builder for the Experiment class
# it will be used to create a set of parameters to grid search over
# and then run the experiments


class ExperimentManager:
    def __init__(self):
        self.fixed_params: Dict = {}
        self.grid_params: Dict[str:List] = {}

        self.queued_experiments: List[Experiment] = []

        # todo: think about how to handle multi-threading

    def build_experiments(self):
        """
        Cleans the list of experiments and builds a new list of Experiment objects
        :return:
        """

        grid_combinations = list(itertools.product(*self.grid_params.values()))
        grid_combinations_dict = [dict(zip(self.grid_params.keys(), combination)) for combination in
                                  grid_combinations]

        experiments = []
        for combination in tqdm.tqdm(grid_combinations_dict):
            experiments.append(Experiment(**combination, **self.fixed_params))

        self.queued_experiments = experiments

    def clone(self):
        return copy.deepcopy(self)

    def baron_solver(self, maxtime: int):
        self.fixed_params['solver'] = 'baron'
        self.fixed_params['tee'] = False
        self.fixed_params['solver_options'] = f'maxtime={maxtime}'

    def random_seed(self):
        # todo: this seed handling is pretty bad
        self.fixed_params['seed'] = random.randint(0, 100000)

    def run_save(self, multi_threaded: bool, n_threads: Optional[int], bar=True, exp_tee=False, accumulate=None):

        iterator = tqdm.tqdm(self.queued_experiments) if bar else self.queued_experiments

        if accumulate is None:
            accumulate = 1

        acc_res = []
        acc_counter = 0

        for exp in iterator:
            if exp_tee:
                print(exp)

            results = exp.run(multi_threaded, n_threads)
            acc_res.extend(results)
            acc_counter += 1
            if acc_counter >= accumulate:
                exp.save_to_disk(acc_res)
                acc_res = []
                acc_counter = 0

        if acc_counter > 0:
            exp.save_to_disk(acc_res)


if __name__ == '__main__':
    nm = ExperimentManager()

    nm.fixed_params['instance_generator'] = random_points_unit_square_with_masses
    nm.fixed_params['instance_arguments'] = {'n': 6}

    nm.baron_solver(10)

    nm.fixed_params['n_runs'] = 10
    nm.fixed_params['save_folder'] = 'runs'

    nm.fixed_params['formulation_arguments'] = {
        'maximum_degree': 3,
        'alpha': .5,
        'use_bind_first_steiner': True,
        'use_convex_hull': True,
        'use_obj_lb': False
    }

    nm.random_seed()
    nm.fixed_params['tee'] = True
    nm.fixed_params['experiment_name'] = 'test_manager'

    nm.grid_params['formulation'] = [
        formulations.dbt.dbt,
    ]

    nm.build_experiments()
    nm.run_save(False, 4)
