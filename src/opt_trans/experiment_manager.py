import copy
from typing import Dict, List, Optional
import itertools

import random

import tqdm

from opt_trans.experiment import Experiment


# this class is a builder for the Experiment class
# that manages multiple runs with different parameters


class ExperimentManager:
    def __init__(self):
        self.fixed_params: Dict = {}
        self.grid_params: Dict[str:List] = {}

        self.queued_experiments: List[Experiment] = []


    def build_experiments(self):
        """
        Builds a list of experiments to run based on the grid parameters and fixed parameters. The experiments are stored
        in the queued_experiments attribute
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
        """
        Sets the solver to baron with a maximum time
        """
        self.fixed_params['solver'] = 'baron'
        self.fixed_params['tee'] = False
        self.fixed_params['solver_options'] = f'maxtime={maxtime} EpsR=0.001'

    def random_seed(self):
        self.fixed_params['seed'] = random.randint(0, 100000)

    def run_save(self, multi_threaded: bool, n_threads: Optional[int], bar=True, exp_tee=False, accumulate=1):
        """
        Runs the experiments and saves the results to disk
        :param multi_threaded: bool
        :param n_threads: Optional[int], ignored if multi_threaded is False
        :param bar: whether to use a tqdm progress bar (stderr)
        :param exp_tee: whether to print experiment details to stdout
        :param accumulate: how many experiment to run before saving to disk
        :return:
        """

        assert len(self.queued_experiments) > 0, "No experiments to run. Call build_experiments first"

        iterator = tqdm.tqdm(self.queued_experiments) if bar else self.queued_experiments

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


