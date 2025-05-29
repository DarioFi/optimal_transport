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
        distances.append(
            (masses[i] ** alpha) * sum((t1[d] - terminals[neg_mass_ind][d]) ** 2 for d in range(len(t1))) ** 0.5)

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


def get_instance_specific_bound(masses, alpha):
    # refers to lemma 4.6 under novel theoretical results

    print(f"Getting instance specific bound for cosine with {alpha=:.2f}")

    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be in [0, 1] for this function.")

    if alpha == 0:
        c = -1 / 2  # sharp bound
    elif alpha <= 1 / 2:
        c = get_instance_specific_bound_lower(masses, alpha)
    elif alpha <= 1:
        c = get_instance_specific_bound_upper(masses, alpha)

    print(f"Instance specific bound is {c:.3f}")

    return c


def get_instance_specific_bound_lower(masses, alpha):
    assert abs(masses[0] + 1) < 1e-5
    assert abs(sum(masses)) < 1e-5
    assert alpha <= 1 / 2
    assert all(masses[i] > 0 for i in range(1, len(masses)))

    m_min = min(masses[1:])

    fa = (1 - m_min ** (2 * alpha) - (1 - m_min) ** (2 * alpha)) / (2 * m_min ** alpha * (1 - m_min) ** alpha)
    fb = ((1 - m_min) ** (2 * alpha) - m_min ** (2 * alpha) - 1) / (2 * m_min ** alpha)

    return max(fa, fb)


import bisect


def get_instance_specific_bound_upper(masses, alpha):
    """
    masses: list of floats, where masses[0] == -1 and sum(masses)==0,
            all masses[1:] > 0.
    alpha:   a float >= 1/2 (not used in r, but asserted for your context).
    returns: the quantity
             r = max_{disjoint A_i,A_j, m_i <= m_j} m_i/m_j.
    """
    assert abs(masses[0] + 1) < 1e-8, "First mass must be -1."
    assert abs(sum(masses)) < 1e-8
    assert alpha >= 0.5
    assert all(m > 0 for m in masses[1:])

    # drop the dummy -1 at index 0
    ms = masses[1:]
    n = len(ms)

    # build a list of (subset_sum, bitmask) for all non-empty subsets
    # mask bit i corresponds to including ms[i]
    subset_sums = []
    for mask in range(1, 1 << n):
        s = 0.0
        # sum up the bits in mask
        for i in range(n):
            if (mask >> i) & 1:
                s += ms[i]
        subset_sums.append((s, mask))

    # sort by sum
    subset_sums.sort(key=lambda x: x[0])
    sums = [s for s, _ in subset_sums]

    r = 0.0
    # we only need to consider A_i with sum ≤ 0.5 (since m_i ≤ m_j and total mass=1)
    for m_i, mask_i in subset_sums:
        if m_i > 0.5:
            break

        # in the sorted list, find the first subset-sum ≥ m_i
        idx = bisect.bisect_left(sums, m_i)

        # scan forward until we find a disjoint A_j
        for m_j, mask_j in subset_sums[idx:]:
            if (mask_i & mask_j) == 0:
                # m_j ≥ m_i by construction
                r = max(r, m_i / m_j)
                break

        # once r == 1 we can't do any better
        if r >= 1.0:
            break

    m_min = min(masses[1:])

    fa = ((1 + r) ** (2 * alpha) - 1 - r ** (2 * alpha)) / (2 * r ** alpha)
    fb = ((1 - m_min) ** (2 * alpha) - m_min ** (2 * alpha) - 1) / (2 * m_min ** alpha)

    return max(fa, fb)


if __name__ == '__main__':
    m = [-1, 0.125, 0.25, 0.25, 0.25, 0.125]

    for alpha in np.linspace(0.1, 1.0, 10):
        print(f"{alpha=:.1f} {get_instance_specific_bound(m, alpha)}")

    print(get_instance_specific_bound(m, 0.001))
