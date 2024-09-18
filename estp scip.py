import math

from pyscipopt import Model, quicksum
import pyscipopt


def euclidean_distance(p1, p2):
    return pyscipopt.sqrt(quicksum((p1[i] - p2[i]) ** 2 for i in range(len(p1))))


def create_scip_model(points_P, initial_steiner_points):
    model = Model("MMX_Euclidean_Steiner")

    P = len(points_P)
    S = len(initial_steiner_points)

    # Variables
    x = {}
    y = {}

    # Create binary variables y_{ij}
    for i in range(P):
        for j in range(S):
            y[i, j] = model.addVar(vtype="B", name=f"y({i},{j})")

    for i in range(S):
        for j in range(i + 1, S):
            y[P + i, j] = model.addVar(vtype="B", name=f"y({P + i},{j})")

    # Create continuous variables x^j for Steiner points
    steiner_vars = {}
    for j in range(S):
        steiner_vars[j] = [
            model.addVar(vtype="C", name=f"x({j},{d})") for d in range(len(initial_steiner_points[0]))
        ]

    obj = model.addVar(vtype="C", name="obj")
    # Objective function: minimize the weighted sum of distances

    model.setObjective(obj, "minimize")

    # IMPORTANT THIS IS AD HOC FOR COUNTEREXAMPLE
    # add constraint that steiner point 0 is at origin
    # for d in range(len(initial_steiner_points[0])):
    #     model.addCons(steiner_vars[0][d] == 0)

    # connect terminal 0 to steiner point 0

    model.addCons(y[0, 0] == 1)
    model.addCons(y[5, 0] == 1)

    model.addCons(y[1, 4] == 1)
    model.addCons(y[6, 4] == 1)

    # Constraint: Each point in P must be connected to exactly one Steiner point in S
    for i in range(P):
        model.addCons(quicksum(y[i, j] for j in range(S)) == 1)

    # Constraint: Each Steiner point must be connected to exactly three edges
    for j in range(S):
        model.addCons(
            quicksum(y[i, j] for i in range(P)) +
            quicksum(y[P + k, j] for k in range(j)) +
            quicksum(y[P + j, k] for k in range(j + 1, S)) == 3
        )

    # Constraint: Connectivity for Steiner points
    for j in range(1, S):
        model.addCons(
            quicksum(y[P + k, j] for k in range(j)) == 1
        )

    fake_obj = model.addCons(quicksum([euclidean_distance(points_P[i], steiner_vars[j]) * y[i, j]
                                       for i in range(P)
                                       for j in range(S)
                                       ] + [
                                          euclidean_distance(steiner_vars[i], steiner_vars[j]) * y[P + i, j]
                                          for i in range(S)
                                          for j in range(i + 1, S)
                                      ]) <= obj)

    return model, y, steiner_vars


if __name__ == "__main__":
    eps = 1e-2
    r = 1

    theta = math.asin((eps / 2) / (r + eps))
    print(math.sin(theta), (eps / 2) / (r + eps))

    x = (r + eps) * math.cos(theta)

    print(math.sqrt(x ** 2 + (eps / 2) ** 2))

    new_x = math.sqrt((r + eps) ** 2 - (eps / 2) ** 2)
    print(new_x, x)

    points_P = [
        (0, 0),
        (r, 0),

        (-eps / 2, x),  # half height vertices
        (r + eps / 2, x),

        (r / 2, x + math.sqrt(3) * (r + eps) / 2),  # top vertex

        # this vertex is theta + 120 from the origin (so bottom left)
        (-(r + eps) * math.cos(theta + math.pi / 3), -(r + eps) * math.sin(theta + math.pi / 3)),

        # this vertex is theta - 120 from the (r,0) (so bottom right)
        ((r + eps) * math.cos(theta - math.pi / 3) + r, (r + eps) * math.sin(theta - math.pi / 3)),
    ]

    # plot points in 2d

    import matplotlib.pyplot as plt

    # also label them with their index
    # plt.scatter([p[0] for p in points_P], [p[1] for p in points_P])

    for i, p in enumerate(points_P):
        plt.scatter(p[0], p[1])
        plt.text(p[0], p[1], str(i))
    plt.show()

    print("My theoretical is: ")
    goal_obj = math.sqrt(3) * (r + eps) * 3

    print(f"{goal_obj=}")


    # compute all distances between points and print matrix

    def act_eucl(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


    import numpy as np

    matrix = [[act_eucl(p1, p2) for p2 in points_P] for p1 in points_P]
    print(np.array(matrix))

    from random import random

    initial_steiner_points = [
        (random(), random()) for _ in range(len(points_P) - 2)
    ]

    model, y, steiner_vars = create_scip_model(points_P, initial_steiner_points)

    model.optimize()

    print("Optimal Solution:")
    if model.getStatus() == "optimal":
        print("Objective value:", model.getObjVal())
        print(
            f"Steiner points: {[model.getVal(steiner_vars[j][d]) for j in range(len(steiner_vars)) for d in range(len(steiner_vars[j]))]}")

        print("Edges:")
        for i in range(len(points_P)):
            for j in range(len(initial_steiner_points)):
                if model.getVal(y[i, j]) > 0.5:
                    print(f"Point {i} connected to Steiner point {j}")

        for i in range(len(initial_steiner_points)):
            for j in range(i + 1, len(initial_steiner_points)):
                if model.getVal(y[len(points_P) + i, j]) > 0.5:
                    print(f"Steiner point {i} connected to Steiner point {j}")

        solution_steiner_points = {
            j: [model.getVal(steiner_vars[j][d]) for d in range(len(steiner_vars[j]))] for j in range(len(steiner_vars))
        }

    # plot the solution
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot steiners
    for j in range(len(initial_steiner_points)):
        ax.scatter(*solution_steiner_points[j], color='r')

    # plot points
    for i in range(len(points_P)):
        ax.scatter(*points_P[i], color='b')

    # plot edges

    for i in range(len(points_P)):
        for j in range(len(initial_steiner_points)):
            if model.getVal(y[i, j]) > 0.5:
                ax.plot([points_P[i][0], solution_steiner_points[j][0]],
                        [points_P[i][1], solution_steiner_points[j][1]],
                        [points_P[i][2], solution_steiner_points[j][2]], color='b')

    for i in range(len(initial_steiner_points)):
        for j in range(i + 1, len(initial_steiner_points)):
            if model.getVal(y[len(points_P) + i, j]) > 0.5:
                ax.plot([solution_steiner_points[i][0], solution_steiner_points[j][0]],
                        [solution_steiner_points[i][1], solution_steiner_points[j][1]],
                        [solution_steiner_points[i][2], solution_steiner_points[j][2]], color='r')

    plt.show()
