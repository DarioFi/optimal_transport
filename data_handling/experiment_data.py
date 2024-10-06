import datetime
import json
import os
from typing import List, Dict, Tuple, Callable, Any
import copy


class C:
    """
    Constraint class
    Usage: C("filter_path") == value returns a constraint object ImplementedC that can later be used to filter data
    """
    SEP = r"//"

    def __init__(self, name):
        self.name = name
        self.filter_path = name.split(C.SEP)

    def __eq__(self, other):
        return ImplementedC(self.filter_path, other, lambda x, y: x == y)

    def __ne__(self, other):
        return ImplementedC(self.filter_path, other, lambda x, y: x != y)

    def __lt__(self, other):
        return ImplementedC(self.filter_path, other, lambda x, y: x < y)

    def __le__(self, other):
        return ImplementedC(self.filter_path, other, lambda x, y: x <= y)

    def __gt__(self, other):
        return ImplementedC(self.filter_path, other, lambda x, y: x > y)

    def __ge__(self, other):
        return ImplementedC(self.filter_path, other, lambda x, y: x >= y)

    def is_in(self, *args):
        return ImplementedC(self.filter_path, args, lambda x, y: x in y)


class ImplementedC:
    """
    Constraint class
    """

    def __init__(self, filter_path, value, test):
        self.filter_path = filter_path
        self.value = value
        self.test = test


class ExperimentData:

    def __init__(self, experiment_name, instance, results, instance_generator, instance_arguments, solver,
                 solver_options, formulation, formulation_arguments, date, seed, cpu):
        self.experiment_name: str = experiment_name
        self.instance: Dict = instance
        self.results: Dict = results
        self.instance_generator: str = instance_generator
        self.instance_arguments: Dict = instance_arguments
        self.solver: str = solver
        self.solver_options: Dict = solver_options
        self.formulation: str = formulation
        self.formulation_arguments: Dict = formulation_arguments
        self.date = date
        self.seed: int = seed
        self.cpu_name: str = cpu

    @classmethod
    def from_json(cls, json_data):
        # Convert the date string back to a datetime object
        date = datetime.datetime.fromisoformat(json_data['date'])
        # Return a new instance of the class with data from json_data
        return cls(
            experiment_name=json_data['experiment_name'],
            instance=json_data['instance'],
            results=json_data['results'],
            instance_generator=json_data['instance_generator'],
            instance_arguments=json_data['instance_arguments'],
            solver=json_data['solver'],
            solver_options=json_data['solver_options'],
            formulation=json_data['formulation'],
            formulation_arguments=json_data['formulation_arguments'],
            date=date,
            seed=json_data['seed'],
            cpu=json_data['cpu']
        )


def exp_is_optimal(exp: ExperimentData):
    if exp.results['termination_condition'] == 'optimal':
        return True
    elif exp.results['termination_condition'] == 'maxTimeLimit':
        return False
    else:
        print(f"Unknown termination condition: {exp.results['termination_condition']}")
        raise ValueError


def NOT(f: Callable[[Any], bool]):
    return lambda x: not f(x)


class Database:
    def __init__(self):
        self.experiments: List[ExperimentData] = []

    @classmethod
    def populate_from_folder(cls, folder):
        db = cls()
        # iterate over json files
        for file in os.listdir(folder):
            if file.endswith(".json"):
                with open(os.path.join(folder, file), 'r') as f:
                    data = json.load(f)
                    assert type(data) == list
                    for run in data:
                        db.experiments.append(ExperimentData.from_json(run))

        return db


def apply_single_C(exp: ExperimentData, c: ImplementedC):
    obj = exp
    for key in c.filter_path:
        if isinstance(obj, dict):
            obj = obj[key]
        else:
            obj = getattr(obj, key)

    return c.test(obj, c.value)


class Query:
    def __init__(self):
        self._filters: List[ImplementedC] = []
        self._callbacks: List[Callable] = []

    def add_filter(self, filter: ImplementedC):
        self._filters.append(filter)
        return self

    def add_callback(self, callback: Callable[[ExperimentData], bool]):
        self._callbacks.append(callback)
        return self

    def clone(self):
        # todo: does it work for callables?
        return copy.deepcopy(self)

    def join(self, other_query):
        # todo:
        # - implement as __add__ ?
        # - return new value or modify self?
        raise NotImplementedError

    def apply(self, db):
        results = []
        for exp in db.experiments:
            if all([filter_func(exp) for filter_func in self._callbacks]):
                if all([apply_single_C(exp, c) for c in self._filters]):
                    results.append(exp)
        return results


if __name__ == '__main__':
    db = Database.populate_from_folder("../runs/")

    query = Query()
    query.add_filter(C("instance_arguments//n").is_in(4,5))
    # query.add_filter(C("results//termination_condition") == "optimal")
    # query.add_filter(C("date") == datetime.datetime.today())

    query_bind = query.clone()
    query_not_bind = query.clone()

    query_bind.add_filter(C("formulation_arguments//bind_first_steiner") == True)
    query_not_bind.add_filter(C("formulation_arguments//bind_first_steiner") == False)

    results_bind = query_bind.apply(db)
    results_not_bind = query_not_bind.apply(db)

    all_exp_bind = [res.results["wallclock_time"] for res in results_bind]
    all_exp_not_bind = [res.results["wallclock_time"] for res in results_not_bind]

    successful_times_bind = [res.results["wallclock_time"] for res in
                             query_bind.clone().add_callback(exp_is_optimal).apply(db)]
    successful_times_not_bind = [res.results["wallclock_time"] for res in
                                 query_not_bind.clone().add_callback(exp_is_optimal).apply(db)]

    failed_times_bind = [res.results["wallclock_time"] for res in
                         query_bind.clone().add_callback(NOT(exp_is_optimal)).apply(db)]

    failed_times_not_bind = [res.results["wallclock_time"] for res in
                             query_not_bind.clone().add_callback(NOT(exp_is_optimal)).apply(db)]

    print(f"Percentage of success with bind: {len(successful_times_bind) / len(all_exp_bind) * 100:.2f}%")
    print(f"Percentage of success without bind: {len(successful_times_not_bind) / len(all_exp_not_bind) * 100:.2f}%")

    for i, (time_bind, time_not_bind) in enumerate(zip(all_exp_bind, all_exp_not_bind)):
        # print with spacing of 10
        time_bind = "XXX" if time_bind > 300 else time_bind
        time_not_bind = "XXX" if time_not_bind > 300 else time_not_bind
        print(f"{i:<10}{time_bind:<10}{time_not_bind:<10}s")
