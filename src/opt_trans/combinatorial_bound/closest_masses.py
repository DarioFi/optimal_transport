import itertools
import random


def sup_theoretical(alpha):
    """Calculate the theoretical upper bound based on alpha."""
    return max(0.0, 2 ** (2 * alpha - 1) - 1)


def cost(m1, m2, alpha):
    """Calculate the cost function between two masses m1 and m2."""
    return ((m1 + m2) ** (2 * alpha) - m1 ** (2 * alpha) - m2 ** (2 * alpha)) / (2 * m1 ** alpha * m2 ** alpha)


def build_sol(x, M):
    """Build the solution for the subset sum problem given x and M."""
    m1, m2 = 0, 0
    for i, m in zip(x, M):
        if i == 1:
            m1 += m
        elif i == 2:
            m2 += m
    return m1, m2


def subset_sum_max_cost(M, alpha):
    """
    Solve the subset sum problem for maximizing the cost function.

    Parameters:
    - M: List of masses.
    - alpha: Coefficient in the cost function

    Returns:
    - The best combination of masses (subset) and the corresponding cost.
    """

    # Generate all possible combinations of indices (0, 1, 2) for each mass in M
    Xs = itertools.product([0, 1, 2], repeat=len(M))

    max_cost_found = float('-inf')
    best_combination = None

    # Iterate over all combinations of subsets
    for x in Xs:
        m1, m2 = build_sol(x, M)
        if m1 == 0 or m2 == 0:  # Skip cases where either subset has no mass
            continue
        c = cost(m1, m2, alpha)
        if c > max_cost_found:
            max_cost_found = c
            best_combination = (x, m1, m2, c)

    return best_combination, max_cost_found, sup_theoretical(alpha)




def best_cost(M, alpha=0.7):
    """Calculate the best cost for a given list of masses M."""
    return subset_sum_max_cost(M, alpha=alpha)[1]


if __name__ == '__main__':
    # Example usage:
    M = [random.random() for _ in range(3)]
    best_combination, max_cost, theoretical_sup = subset_sum_max_cost(M, alpha=0.7)

    print(f"Masses: {M}")
    print(f"Best combination: {best_combination}")
    print(f"Maximum cost found: {max_cost}")
    print(f"Theoretical upper bound: {theoretical_sup}")
