import datetime
import json
import os
import random
from typing import List, Dict, Callable

from pyomo.environ import Var
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from formulations.mmx import mmx_model
from formulations.gmmx import gmmx
from problems.closest_counterexample import random_points_unit_square, random_points_unit_square_with_masses, \
    fixed_points


def extract_results(model, result):
    results_dict = {
        'status': str(result.solver.status),
        'termination_condition': str(result.solver.termination_condition),
        'objective': model.obj(),
        'lower_bound': result.Problem[0]["Lower bound"],
        'upper_bound': result.Problem[0]["Upper bound"],
        'wallclock_time': result.Problem[0]["Wall time"],
        'time': result.solver.time,
        # 'variables': {v.name: pyo.value(model.__getattribute__(v.name)) for v in model.component_objects(Var)}
    }
    vars_dict = {}
    for x in model.component_objects(Var):
        indices = dict(x)
        vars_dict[x.name] = {}
        z = vars_dict[x.name]
        for key, value in indices.items():
            z[str(key)] = float(pyo.value(value))
    results_dict['variables'] = vars_dict
    return results_dict


class Experiment:
    def __init__(self, instance_generator: Callable, instance_arguments: Dict, solver: str, solver_options: str,
                 formulation: Callable,
                 formulation_arguments: Dict, seed: int, save_folder: str, experiment_name: str):
        self._instance_generator = instance_generator
        self.instance_arguments = instance_arguments

        self.solver = solver
        self.solver_options = solver_options

        self.formulation = formulation
        self.formulation_arguments = formulation_arguments

        self.seed = seed

        self.save_location = save_folder
        self.experiment_name = experiment_name

        self.cpu_name = os.popen("lscpu | grep 'Model name'").read().split(":")[1].strip()

    def _single_run(self, seed):
        random.seed(seed)

        instance = self._instance_generator(**self.instance_arguments)
        formulation = self.formulation(**instance, **self.formulation_arguments)

        solver = SolverFactory(self.solver)

        results = solver.solve(formulation, tee=True, options_string=self.solver_options)

        results_serializable = extract_results(formulation, results)
        return instance, results_serializable

    def serialize(self, instance, results, seed):
        """
        Build a dictionary with the instance, formulation, results and all the parameters of the experiment
        :param instance:
        :param formulation:
        :param results:
        :return:
        """
        # todo: extract solution from results
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

    def run(self, n_runs: int):
        random.seed(self.seed)
        seeds = [random.randint(0, 100000) for _ in range(n_runs)]
        results = []
        for seed in seeds:
            instance, result = self._single_run(seed)
            results.append(self.serialize(instance, result, seed))

        return results

    def save_to_disk(self, results: List[Dict]):
        with open(f'{self.save_location}/{datetime.datetime.now().isoformat()}_{self.experiment_name}.json', 'w') as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    for bind in [True, False]:
        exp = Experiment(
            instance_generator=random_points_unit_square_with_masses,
            instance_arguments={'n': 4},
            solver='baron',
            solver_options='maxtime=300',
            formulation=gmmx,
            formulation_arguments={
                'maximum_degree': 3,
                'alpha': .5,
                'bind_first_steiner': bind,
                'use_obj_lb': True
            },
            seed=83810,
            save_folder='runs',
            experiment_name=f'gmmx_with_{bind=}'
        )

        results = exp.run(50)
        exp.save_to_disk(results)

# todo:
# experiment manager that runs multiple experiments with grid search or similar
# maybe parallelize the experiments
# create deserializer tool to read the results and analyze them
# visualization
# install Gurobi on the VM
# install CPLEX on the VM
