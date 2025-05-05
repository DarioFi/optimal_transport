from closest_masses import best_cost
import tqdm
import random
import numpy as np

eps = 1e-4


def gen_random_masses(n, seed):
    random.seed(seed)
    l = [random.random() for _ in range(n)]
    s = sum(l)
    return [x / s for x in l]


data = []

alphas = np.linspace(0 + eps, 1 - eps, 10)
for alpha in tqdm.tqdm(alphas):
    for n in range(2, 11, 1):
        for seed in range(50):
            masses = gen_random_masses(n, seed)
            m = min(masses)
            M = 1
            # raise NotImplemented("We should solve the subset problem sum here")
            # cos_theta = (M + m) ** (2 * alpha) - m ** (2 * alpha) - M ** (2 * alpha)
            # cos_theta /= 2 * (m * M) ** alpha
            cos_theta = best_cost(masses, alpha)
            data.append((n, alpha, cos_theta))


import json

with open("closest_masses.json", "w") as f:
    json.dump(data, f)