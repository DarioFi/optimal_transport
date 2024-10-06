import datetime
import json
import os
from typing import List


class ExperimentData:

    def __init__(self, experiment_name, instance, results, instance_generator, instance_arguments, solver,
                 solver_options, formulation, formulation_arguments, date, seed, cpu):
        self.experiment_name = experiment_name
        self.instance = instance
        self.results = results
        self.instance_generator = instance_generator
        self.instance_arguments = instance_arguments
        self.solver = solver
        self.solver_options = solver_options
        self.formulation = formulation
        self.formulation_arguments = formulation_arguments
        self.date = date
        self.seed = seed
        self.cpu_name = cpu

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

    def is_optimal(self):
        return self.results['termination_condition'] == 'optimal'


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


if __name__ == '__main__':
    db = Database.populate_from_folder("../runs/")

    successful_times_bind = []
    successful_times_not_bind = []
    failed_times_bind = []
    failed_times_not_bind = []
    for exp in db.experiments:
        print(exp.results)
        if exp.is_optimal():
            if exp.formulation_arguments["bind_first_steiner"]:
                successful_times_bind.append(exp.results["wallclock_time"])
            else:
                successful_times_not_bind.append(exp.results["wallclock_time"])

        else:
            if exp.formulation_arguments["bind_first_steiner"]:
                failed_times_bind.append(exp.results["wallclock_time"])
            else:
                failed_times_not_bind.append(exp.results["wallclock_time"])


    print(f"Successful times bind: {successful_times_bind}")
    print(f"Failed times bind: {failed_times_bind}")

    print(f"Average successful times bind: {sum(successful_times_bind)/len(successful_times_bind)}")

    print(f"Successful times not bind: {successful_times_not_bind}")
    print(f"Failed times not bind: {failed_times_not_bind}")

    print(f"Average successful times not bind: {sum(successful_times_not_bind)/len(successful_times_not_bind)}")
