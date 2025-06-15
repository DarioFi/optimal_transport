def points_normalization(points):
    """
    Point normalization function.
    """
    M = max(norm(x, y, range(len(points[0]))) for x in points for y in points if x != y)
    points = [[x[j] / M for j in range(len(x))] for x in points]
    return points


def norm(p1, p2, dim):
    return sum((p1[d] - p2[d]) ** 2 for d in dim) ** 0.5

