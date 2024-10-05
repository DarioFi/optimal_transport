import datetime


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


class Database:
    def __init__(self, folder):
        self.experiments = []

    def populate(self, folder):
        # iterate over json in folder
        # load them with experiment constructor
        pass


