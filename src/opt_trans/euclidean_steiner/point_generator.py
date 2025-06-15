# utilities for steiner graph ILP

import random


def gen_all(n_points, n_edges, n_terminals, dim):
    points = list(range(n_points))
    pt_c = {}
    while len(pt_c) < n_points:
        w = [random.uniform(0, 10) for _ in range(dim)]
        if w not in pt_c.values():
            pt_c[len(pt_c)] = w
    terminals = random.sample(points, n_terminals)

    all_edges = [(i, j) for i in points for j in points if i < j]
    edges = random.sample(all_edges, n_edges)

    return points, pt_c, terminals, edges


def reduce_to_convex_hull(points, pt_c, terminals, edges):
    hull_points = [pt_c[t] for t in terminals]

    # test if points are in covex hull using geometry library
    from scipy.spatial import Delaunay

    if not isinstance(hull_points, Delaunay):
        hull = Delaunay(hull_points)

    keep_p = []
    for point in points:
        if point in terminals:
            keep_p.append(point)
            continue
        coord = pt_c[point]
        if hull.find_simplex(coord) >= 0:
            keep_p.append(point)

    keep_edges = []
    for edge in edges:
        if edge[0] in keep_p and edge[1] in keep_p:
            keep_edges.append(edge)

    return keep_p, pt_c, terminals, keep_edges
