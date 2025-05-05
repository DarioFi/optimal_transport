import itertools

import pytest
from pyomo.opt import SolverFactory

from formulations.gmmx import gmmx
from formulations.dbt import dbt


@pytest.mark.parametrize("formulation", [gmmx, dbt])
def test_3_terminals(formulation):
    terminals = [(0, 1), (1, 0), (1, 1)]
    masses = [-1, .5, .5]
    maximum_degree = 3
    alpha = .5

    instance = formulation(terminals, masses, maximum_degree, alpha, use_bind_first_steiner=False, use_obj_lb=False,
                           use_convex_hull=False)
    solver = SolverFactory('baron')

    results = solver.solve(instance, tee=True)

    assert results.solver.termination_condition == 'optimal'


def test_consistency_use():
    terminals = [(0, 1), (1, 0), (1, 1)]
    masses = [-1, .5, .5]
    maximum_degree = 3
    alpha = .5

    uses = itertools.product([True, False], repeat=3)

    lbs = []
    ubs = []

    for use_bind_first_steiner, use_obj_lb, use_convex_hull in uses:
        instance = dbt(terminals, masses, maximum_degree, alpha, use_bind_first_steiner, use_obj_lb, use_convex_hull)
        solver = SolverFactory('baron')

        results = solver.solve(instance, tee=True)

        lbs.append(results.problem.lower_bound)
        ubs.append(results.problem.upper_bound)

    assert abs(max(lbs) - min(lbs)) < 1e-6
    assert abs(max(ubs) - min(ubs)) < 1e-6
