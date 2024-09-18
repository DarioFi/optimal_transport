from problems.utils import points_normalization
from problems.closest_counterexample import get_5_points_eps_0_1

def example():
    points = [(0, 0), (0, 1), (1, 0), (1, 1), (.5, .5)]

    masses = [-1, .9, -1, .3, .3]

    positives = sum([m for m in masses if m > 0])
    negatives = -sum([m for m in masses if m < 0])

    masses_w = [m / positives if m > 0 else m / negatives for m in masses]

    points = points_normalization(points)

    return points, masses_w


def get_5_masses():
    points, _ = get_5_points_eps_0_1()

    masses = [-1, .5, -1, .3, .6]

    positives = sum([m for m in masses if m > 0])
    negatives = -sum([m for m in masses if m < 0])

    masses_w = [m / positives if m > 0 else m / negatives for m in masses]

    return points, masses_w