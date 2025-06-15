import datetime
import json
import os
import random
from typing import List, Dict, Callable, Optional

from pyomo.environ import Var
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, UndefinedData


def extract_results(model, result, solver) -> Dict:
    """
    Extract the results from a pyomo model and a solver result
    """
    try:
        obj = model.obj()
    except ValueError:
        obj = None

    try:
        iterations = result.Problem[0]["Iterations"]
    except:
        iterations = None

    results_dict = {
        'status': str(result.solver.status),
        'termination_condition': str(result.solver.termination_condition),
        'objective': obj,
        'lower_bound': result.Problem[0]["Lower bound"],
        'upper_bound': result.Problem[0]["Upper bound"],
        'iterations': iterations,

    }


    if 'gurobi' in solver or 'baron' not in solver:
        time = result.solver[0]["System time"]
        if not isinstance(time, UndefinedData):
            results_dict['time'] = time
        else:
            results_dict['time'] = None

        results_dict['wallclock_time'] = result.solver[0]["Wallclock time"]

        if isinstance(results_dict['wallclock_time'], UndefinedData):
            results_dict['wallclock_time'] = None
    else:
        results_dict['time'] = result.Problem[0]["cpu time"]
        results_dict['wallclock_time'] = result.Problem[0]["Wall time"]

        if isinstance(results_dict['wallclock_time'], UndefinedData):
            results_dict['wallclock_time'] = None

    vars_dict = {}
    for x in model.component_objects(Var):
        indices = dict(x)
        vars_dict[x.name] = {}
        z = vars_dict[x.name]
        for key, value in indices.items():
            try:
                z[str(key)] = float(pyo.value(value))
            except ValueError:
                pass

    results_dict['variables'] = vars_dict
    return results_dict


class Experiment:
    """
    This class defines a single experiment, consisting of instance generators, formulations, solver, solver options, seed
    and number of runs. It can run (in parallel too) and save the results to disk.
    """

    def __init__(self, instance_generator: Callable, instance_arguments: Dict, solver: str, solver_options: str,
                 formulation: Callable, n_runs: int,
                 formulation_arguments: Dict, seed: int, save_folder: str, experiment_name: str, tee: bool):
        self._instance_generator = instance_generator
        self.instance_arguments = instance_arguments

        self.solver = solver
        self.solver_options = solver_options
        self.tee = tee

        self.formulation = formulation
        self.formulation_arguments = formulation_arguments

        assert "alpha" not in self.formulation_arguments

        self.seed = seed

        self.n_runs = n_runs
        self.save_location = save_folder
        self.experiment_name = experiment_name

        self.cpu_name = os.popen("lscpu | grep 'Model name'").read().split(":")[1].strip()

    def _single_run(self, seed):
        random.seed(seed)

        instance = self._instance_generator(**self.instance_arguments)
        formulation = self.formulation(**instance, **self.formulation_arguments)

        solver = SolverFactory(self.solver)

        if self.solver == "gurobi_persistent":
            # For persistent solvers, set the instance first
            solver.set_instance(formulation)
            results = solver.solve(tee=self.tee, options=self.solver_options)
        else:
            results = solver.solve(formulation, tee=self.tee, options_string=self.solver_options)

        results_serializable = extract_results(formulation, results, self.solver)

        return instance, results_serializable

    def serialize(self, instance, results, seed) -> Dict:
        """
        Build a dictionary with the instance, formulation, results and all the parameters of the experiment
        """

        objective = results["objective"]

        if isinstance(objective, complex):
            assert abs(objective.imag) < 1e-5
            results["objective"] = objective.real

        serialized_data = {
            'experiment_name': self.experiment_name,
            'instance': instance,
            'results': results,
            'instance_generator': self._instance_generator.__name__,
            'instance_arguments': self.instance_arguments,
            'solver': self.solver,
            'solver_options': self.solver_options,
            'formulation': self.formulation.__name__,
            'formulation_arguments': self.formulation_arguments,
            'date': datetime.datetime.now().isoformat(),
            'seed': seed,
            'cpu': self.cpu_name
        }

        return serialized_data

    def run(self, multithreaded: bool, n_threads: Optional[int] = None) -> List[Dict]:
        """
        Run the experiment n_runs times calling _single_run each time
        :param multithreaded: whether to spawn multiple threads using multiprocessing
        :param n_threads: number of threads to spawn, ignored if multithreaded is False
        :return:
        """
        random.seed(self.seed)
        seeds = [random.randint(0, 100000) for _ in range(self.n_runs)]
        results = []
        if multithreaded is False:
            for seed in seeds:
                instance, result = self._single_run(seed)
                results.append(self.serialize(instance, result, seed))
        else:
            import multiprocessing

            if n_threads is None:
                n_threads = multiprocessing.cpu_count()

            with multiprocessing.Pool(n_threads) as pool:
                single_runs_res = pool.map(self._single_run, seeds)

            for seed, (instance, result) in zip(seeds, single_runs_res):
                results.append(self.serialize(instance, result, seed))

        return results

    def save_to_disk(self, results: List[Dict]):
        """
        Save the results to disk in a json file
        """
        if not os.path.exists(self.save_location):
            os.makedirs(self.save_location)

        with open(f'{self.save_location}/{datetime.datetime.now().isoformat()}_{self.experiment_name}.json', 'w') as f:
            json.dump(results, f, indent=4)

    def __str__(self):
        """Print all the parameters of the experiment"""
        s = ""
        s += f"Experiment: {self.experiment_name}\n"
        s += f"Instance generator: {self._instance_generator.__name__}\n"
        s += f"Instance arguments: {self.instance_arguments}\n"
        s += f"Solver: {self.solver}\n"
        s += f"Solver options: {self.solver_options}\n"
        s += f"Formulation: {self.formulation.__name__}\n"
        s += f"Formulation arguments: {self.formulation_arguments}\n"
        s += f"Number of runs: {self.n_runs}\n"
        s += f"Seed: {self.seed}\n"
        return s
