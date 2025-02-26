from data_handling.experiment_data import ExperimentData, Database, Query, C
import matplotlib.pyplot as plt


def visualize(data: ExperimentData):
    terminals = data.instance['terminals']
    masses = data.instance['masses']
    alpha = data.instance['alpha']

    plt.figure()
    plt.title(data.experiment_name)
    plt.axis('equal')
    plt.axis('off')

    steiner_points = []
    x = data.results['variables']['x']
    polished_x = {}
    for key, value in x.items():
        # key = "(0,1,2)"
        # transform it into a tuple of integers
        new_k = tuple(map(int, key[1:-1].split(',')))

        polished_x[new_k] = value

    # polished_x = {
    # (x_0, coordinate) : value
    # (x_0, coordinate) : value
    # ...
    # }

    # transform into x = { 0: (coordinate1, coordinate2), 1: (coordinate1, coordinate2), ...}

    new_x = {}
    for key, value in polished_x.items():
        if key[0] not in new_x:
            new_x[key[0]] = []
        new_x[key[0]].append(value)

    steiner_points = new_x

    print(f"{steiner_points=}")

    for point, value in steiner_points.items():
        plt.plot(value[0], value[1], 'go')

    try:
        flows = data.results['variables']['f']

    except:
        flows = data.results['variables']['y']

    new_flow = {}

    for key, value in flows.items():
        new_k = tuple(map(int, key[1:-1].split(',')))
        new_flow[new_k] = value

    all_points_indexed = terminals + list(steiner_points.values())

    cost = 0

    for key, value in new_flow.items():
        if value > 0:
            value = min(value, 1)

            distance = ((all_points_indexed[key[0]][0] - all_points_indexed[key[1]][0]) ** 2 + (
                    all_points_indexed[key[0]][1] - all_points_indexed[key[1]][1]) ** 2) ** 0.5
            cost += value ** alpha * distance

            value = value ** .3

            plt.plot([all_points_indexed[key[0]][0], all_points_indexed[key[1]][0]],
                     [all_points_indexed[key[0]][1], all_points_indexed[key[1]][1]], 'b-', alpha=value)

            # clip value to 1
            # do the same but add an arrow at the end

            # plt.arrow(all_points_indexed[key[0]][0],
            #           all_points_indexed[key[0]][1],
            #           all_points_indexed[key[1]][0] - all_points_indexed[key[0]][0],
            #           all_points_indexed[key[1]][1] - all_points_indexed[key[0]][1],
            #           head_width=0.05, head_length=0.02,
            #           fc='b', ec='b', alpha=value**.2)

    for mass, terminal in zip(masses, terminals):
        # terminal is a list with x,y coordinates, mass is the intensity
        color = 'ro' if mass < 0 else 'bo'
        plt.plot(terminal[0], terminal[1], color, alpha=1)

    print(f"Cost is:  {cost}")
    # todo : fix colors overlapping
    plt.show()


if __name__ == '__main__':
    db = Database.populate_from_folder("../gurobi_test_thrash/")

    query = Query()

    exps = query.apply(db)

    visualize(exps[0])
    visualize(exps[1])
    visualize(exps[2])
    visualize(exps[3])
    visualize(exps[4])
    visualize(exps[5])
