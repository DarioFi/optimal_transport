# we weant to compare the runtime performance of different baseline formulations for multiple n
# this test is done at alpha = 0

from opt_trans.experiment_manager import ExperimentManager
from opt_trans.problems.instance_generators import random_points_unit_square
from opt_trans.formulations.dbt import dbt_alpha_0
