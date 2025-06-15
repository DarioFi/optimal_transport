import datetime
import itertools
import json
import os
from typing import List, Dict, Tuple, Callable, Any
import copy
import numpy as np
from pyomo.contrib.mindtpy.util import numpy


def is_hashable(obj):
    try:
        hash(obj)
    except TypeError:
        return False
    return True


def recursive_freeze(obj):
    """
    Utility for making obj a hashable object, by recursively building frozen sets.
    """
    if is_hashable(obj):
        return obj
    if isinstance(obj, dict):
        return frozenset({k: recursive_freeze(v) for k, v in obj.items()}.items())
    try:
        return tuple(recursive_freeze(v) for v in obj)
    except TypeError:
        raise TypeError(f"Object {obj} is weird")


class C:
    """
    Constraint class
    Usage: C("filter_path") == value returns a constraint object ImplementedC that can later be used to filter data
    Used for writing filter queries in a concise simple way in the ExperimentData and Database class.
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


class ImplementedC(C):
    """
    Auxiliary class that represents an implemented constraint, built from a constraint C
    """

    def __init__(self, filter_path, value, test, name=None):
        self.filter_path = filter_path
        self.value = value
        self.test = test


class ExperimentData:
    """
    This class represents the result of a single experiment, comprising of all the useful information that can be carried.
    It can automatically load form the json serialized by Experiment and ExperimentManager classes.
    """
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


def NOT(f: Callable[[Any], bool]):
    """
    Auxiliary function for writing constraints
    """
    return lambda x: not f(x)


class Database:
    """
    This class groups ExperimentData objects with and underlying list, and exposes methods to populate the db from a folder
    and perform queries / indexing on specific columns of the db.
    """
    def __init__(self):
        self.experiments: List[ExperimentData] = []

    @classmethod
    def populate_from_folder(cls, folder):
        """
        Populate experiments from a folder by reading all json files.
        """

        db = cls()
        for file in os.listdir(folder):
            if file.endswith(".json"):
                with open(os.path.join(folder, file), 'r') as f:
                    data = json.load(f)
                    assert type(data) == list
                    for run in data:
                        db.experiments.append(ExperimentData.from_json(run))

        return db

    def __len__(self):
        return len(self.experiments)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.experiments[item]
        if isinstance(item, int):
            return self.experiments[item]
        if isinstance(item, str):
            db = Database()
            for exp in self.experiments:
                if isinstance(exp, dict):
                    if item in exp:
                        db.experiments.append(exp[item])
                    else:
                        raise KeyError(f"Key {item} not found in experiment")
                else:
                    db.experiments.append(getattr(exp, item))

            return db

    def to_numpy(self):
        return numpy.array(self.experiments)

    def index_on(self, *args, ) -> Dict:
        """
        Transforms the db from an unstructured list of ExperimentData to a Dict[ . : List[ExperimentData] ]
        :param args: list of strings or constraints C. The indexing will be performed jointly an all the args passed
        """
        new_db = {}

        for exp in self.experiments:

            key = []
            for arg in args:
                if isinstance(arg, C):
                    key.append(arg.name)
                    key.append(extract_C(exp, arg))
                else:
                    key.append(arg)
                    key.append(getattr(exp, arg))

            key = recursive_freeze(key)

            if key in new_db:
                new_db[key].append(exp)
            else:
                new_db[key] = [exp]

        return new_db


def extract_C(exp: ExperimentData, c: C):
    """
    Access experiment data and extracts the relevant data for the constraint C
    """

    obj = exp
    for key in c.filter_path:
        if isinstance(obj, dict):
            if key not in obj:
                return False
            obj = obj[key]
        else:
            obj = getattr(obj, key)

    return obj


def apply_single_C(exp: ExperimentData, c: ImplementedC) -> bool:
    """
    Returns a boolean of whether the constraint C is satisfied by exp
    """
    obj = extract_C(exp, c)
    return c.test(obj, c.value)


class Query:
    """
    Builder class that joins multiple constraints that can then be applied to a database
    """

    def __init__(self):
        self._filters: List[ImplementedC] = []
        self._callbacks: List[Callable] = []

    def add_filter(self, filter: ImplementedC):
        self._filters.append(filter)
        return self

    def add_callback(self, callback: Callable[[ExperimentData], bool]):
        """
        General filter that applies the callable to all experiments and keeps only the ones with a true value
        """
        self._callbacks.append(callback)
        return self

    def clone(self):
        return copy.deepcopy(self)

    def apply(self, db: Database):
        """
        Applies to a database and returns a new Database
        Warning: experiments are passed by reference so modifying them might create issues
        """
        results = Database()
        for exp in db.experiments:
            if all([filter_func(exp) for filter_func in self._callbacks]):
                if all([apply_single_C(exp, c) for c in self._filters]):
                    results.experiments.append(exp)
        return results


if __name__ == '__main__':
    db = Database.populate_from_folder("../runs/")

    # example usage 1

    query = Query()
    # query.add_filter(C("results//termination_condition") == "optimal")
    # query.add_filter(C("date") == datetime.datetime.today())

    query.add_filter(C("instance_arguments//n") == 6)
    query.add_filter(C("formulation") == "dbt")

    query_optimal = query
    query_optimal.add_filter(C("results//termination_condition") == "optimal")

    results_optimal = query_optimal.apply(db)

    print(f"Optimal: {len(results_optimal)}")

    times_opt = results_optimal["results"]["time"]

    # filter on use_bind_steiner and use_convex_hull to get a 2x2 matrix of results

    conds = itertools.product([True, False], repeat=2)
    data = {}
    for co in conds:
        nq = query_optimal.clone()
        data[tuple(co)] = nq.add_filter(
            C("formulation_arguments//use_bind_first_steiner") == co[0]).add_filter(
            C("formulation_arguments//use_convex_hull") == co[1]
        ).apply(db)["results"]["time"].to_numpy()

    mean = np.array([np.mean(x) for x in data.values()]).reshape(2, 2)
    std = np.array([np.std(x) for x in data.values()]).reshape(2, 2)

    print(f"Average time: {sum(times_opt) / len(times_opt)}")
    print(f"Max time: {max(times_opt)}")
    print(f"Min time: {min(times_opt)}")

    print(mean.round(2))
    print(std.round(2))

    for k, d in data.items():
        print(rf"bind_first={k[0]} convex_hull={k[1]}: {np.mean(d):2f} +- {np.std(d):2f}")

    # ------------------------------------------------
    # example usage 2:

    q = Query().add_filter(C("instance_arguments//n") == 7).add_filter(C("formulation") == "dbt")
    db = q.apply(db)

    data = db.index_on(C("formulation_arguments//use_better_obj"), C("instance_arguments//n"))

    print(len(data))


    def avg_time(exps):
        return sum([exp.results["time"] for exp in exps]) / len(exps)


    for k, v in data.items():
        print(k, avg_time(v))
