import math
from opt_trans.problems.utils import norm
from random import random
from utils import points_normalization
import sympy as sp


def get_7_points(eps):
    """
    Counterexample construction with 7 points. It can be refined to 5 points as shown in the thesis and below.
    :param eps:
    :return:
    """
    r = 1
    theta = math.asin((eps / 2) / (r + eps))
    x = (r + eps) * math.cos(theta)
    terminals = [
        (0, 0),
        (r, 0),
        (-eps / 2, x),  # half-height vertices
        (r + eps / 2, x),
        (r / 2, x + math.sqrt(3) * (r + eps) / 2),  # top vertex
        (-(r + eps) * math.cos(math.pi / 2 - theta - math.pi / 3),
         +(r + eps) * math.sin(math.pi / 2 - theta - math.pi / 3)),
        ((r + eps) * math.cos(math.pi / 2 - theta - math.pi / 3) + r,
         +(r + eps) * math.sin(math.pi / 2 - theta - math.pi / 3)),
    ]

    terminals = points_normalization(terminals)

    return {"terminals": terminals}


def get_5_points_computations(eps):
    """
    Counterexample construction with 5 points for the structural results section of the thesis.
    :param eps: how much closer are the closest points compared to the equilateral triangles
    :return:
    """
    r = 1
    theta = math.asin((eps / 2) / (r + eps))
    x = (r + eps) * math.cos(theta)

    points = [
        (0, 0),
        (-eps / 2, x),  # half-height vertices
        (-(r + eps) * math.cos(math.pi / 2 - theta - math.pi / 3),
         +(r + eps) * math.sin(math.pi / 2 - theta - math.pi / 3)),
    ]

    # Find intersection of circle centered in points[1] with radius r + eps and circle centered in points[0] with radius r
    x, y = sp.symbols('x y')
    eq1 = (x - points[0][0]) ** 2 + (y - points[0][1]) ** 2 - r ** 2
    eq2 = (x - points[1][0]) ** 2 + (y - points[1][1]) ** 2 - (r + eps) ** 2
    sol = sp.solve([eq1, eq2], (x, y))
    act_sol = sol[1]
    points.append((act_sol[0], act_sol[1]))

    # Intersection of circles of radius r + eps from points[1] and points[3]
    eq1 = (x - points[1][0]) ** 2 + (y - points[1][1]) ** 2 - (r + eps) ** 2
    eq2 = (x - points[3][0]) ** 2 + (y - points[3][1]) ** 2 - (r + eps) ** 2
    sol = sp.solve([eq1, eq2], (x, y))
    act_sol = sol[1]
    points.append((act_sol[0], act_sol[1]))

    # Make sure points are floats
    points = [[float(x) for x in p] for p in points]

    points = points_normalization(points)

    return {"terminals": points}


def get_5_points_eps_0_1():
    """
    Cached result for the 5 points counterexample with eps = 0.1.
    :return:
    """
    points = [[0.0, 0.0], [-0.022757723856420418, 0.5001524364085584],
              [-0.4445235776227028, 0.23036745121230792],
              [0.39559375470752195, 0.22510259724498904],
              [0.4246181634480259, 0.7249305249739288]]
    M = max(norm(x, y, range(len(points[0]))) for x in points for y in points if x != y)

    points = [[x[j] / M for j in range(len(x))] for x in points]

    # Make sure points are floats
    points = [[float(x) for x in p] for p in points]

    return {"terminals": points}


def regular_nagon(n=7):
    """
    Generate points for regular n-agon centered in 0,0
    """
    r = 1
    points = [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)) for i in range(n)]
    M = max(norm(x, y, range(len(points[0]))) for x in points for y in points if x != y)
    points = [[x[j] / M for j in range(len(x))] for x in points]

    return {"terminals": points}


def hypercube(dim=2):
    """
    Generate points for a hypercube in dimension dim.
    """
    points = []
    for i in range(2 ** dim):
        point = []
        for j in range(dim):
            point.append((i >> j) & 1)
        points.append(point)

    M = max(norm(x, y, range(len(points[0]))) for x in points for y in points if x != y)
    points = [[x[j] / M for j in range(len(x))] for x in points]

    return {"terminals": points}



def random_points_unit_square_with_masses(n, alpha=0):
    """
    Generate n random points in the unit square with masses, all uniformly distributed in [0, 1].
    Returns a dictionary compatible with the formulation functions.
    """

    points = [[random(), random()] for _ in range(n)]

    masses = [random() for _ in range(n - 1)]

    # renormalize masses to 1
    masses = [m / sum(masses) for m in masses]

    masses = [-sum(masses)] + masses

    return {"terminals": points, "masses": masses, "alpha": alpha}


def random_points_unit_cube_with_masses(n, dim=2, alpha=0):
    """
    Generates random points, but supports also higher dimensions.
    """
    points = [[random() for _ in range(dim)] for _ in range(n)]

    masses = [random() for _ in range(n - 1)]
    # renormalize masses to 1
    masses = [m / sum(masses) for m in masses]
    masses = [-sum(masses)] + masses

    return {"terminals": points, "masses": masses, "alpha": alpha}


