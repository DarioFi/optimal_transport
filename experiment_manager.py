from typing import Callable, Dict

from experiment import Experiment


class ExperimentManager:
    def __init__(self,
                 instance_generator: Callable, instance_arguments: Dict, solver: str, solver_options: str,
                 formulation: Callable, formulation_arguments: Dict, seed: int, save_folder: str, experiment_name: str
                 ):
        pass
