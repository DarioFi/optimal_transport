from typing import List

import numpy as np


def get_maximum_degree(M: List[float]) -> float:
    raise NotImplementedError


def get_lower_bound(terminals, masses, alpha):
    """
    Check that there is only one negative mass. Compute the distance with every other terminal weighted by mass**alpha * distance
    Then take the minimum
    :param terminals:
    :param masses:
    :return:
    """

    neg_mass_ind = None
    # check negative mass
    for i, m in enumerate(masses):
        if m < 0:
            if neg_mass_ind is not None:
                # too many negative masses
                return 0
            else:
                neg_mass_ind = i

    # compute distances
    distances = []
    for i, t1 in enumerate(terminals):
        if i == neg_mass_ind:
            continue
        distances.append((masses[i] ** alpha) * sum((t1[d] - terminals[neg_mass_ind][d]) ** 2 for d in range(len(t1))) ** 0.5)

    return min(distances)


if __name__ == '__main__':
    M = [-1, 1 / 3, 1 / 3, 1 / 3]
    terminals = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ]

    print(get_lower_bound(terminals, M, 1))
