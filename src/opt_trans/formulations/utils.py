import numpy as np


def get_lower_bound(terminals, masses, alpha):
    """
    Lower bound for the optimal value of the problem, used in the gmmx formulation.
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


def get_instance_specific_bound(masses, alpha):
    """
    Computes the results of lemma 4.6 from the thesis, under the theoretical results section.
    """

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
    """
    Computes the lower bound in the case of alpha <= 1/2.
    """
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
    Computes the upper bound in the case of alpha > 1/2.
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

        for i in range(n):
            if (mask >> i) & 1:
                s += ms[i]
        subset_sums.append((s, mask))

    subset_sums.sort(key=lambda x: x[0])
    sums = [s for s, _ in subset_sums]

    r = 0.0
    # we only need to consider A_i with sum ≤ 0.5 (since m_i ≤ m_j and total mass=1)
    for m_i, mask_i in subset_sums:
        if m_i > 0.5:
            break

        # in the sorted list, find the first subset-sum ≥ m_i
        idx = bisect.bisect_left(sums, m_i)

        for m_j, mask_j in subset_sums[idx:]:
            if (mask_i & mask_j) == 0:
                r = max(r, m_i / m_j)
                break

        if r >= 1.0:
            break

    m_min = min(masses[1:])

    fa = ((1 + r) ** (2 * alpha) - 1 - r ** (2 * alpha)) / (2 * r ** alpha)
    fb = ((1 - m_min) ** (2 * alpha) - m_min ** (2 * alpha) - 1) / (2 * m_min ** alpha)

    return max(fa, fb)


def get_general_cosine_bound(alpha):
    """
    Instance-agnostic bound
    """
    return max(0, 2 ** (2 * alpha - 1) - 1)
