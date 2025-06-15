# Old script that visualizes with a minimal style the optimal solution as a graph in R2


from opt_trans.data_handling.experiment_data import ExperimentData, Database, Query, C
import matplotlib.pyplot as plt


def visualize(data: ExperimentData):
    terminals = data.instance['terminals']
    masses = data.instance['masses']
    alpha = data.instance['alpha']

    plt.figure()
    # plt.title(data.experiment_name)
    plt.axis('equal')
    # plt.axis('off')

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

    is_lab_st = False
    for point, value in steiner_points.items():
        lab = None if is_lab_st else "Steiner"
        is_lab_st = True
        plt.plot(value[0], value[1], 'go', label=lab)

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
    flat_length = 0
    for key, value in new_flow.items():
        if value > 1e-8:
            value = min(value, 1)

            distance = ((all_points_indexed[key[0]][0] - all_points_indexed[key[1]][0]) ** 2 + (
                    all_points_indexed[key[0]][1] - all_points_indexed[key[1]][1]) ** 2) ** 0.5
            cost += value ** alpha * distance
            flat_length += distance
            print(key, value)
            print(distance)

            value = value ** .8

            plt.plot([all_points_indexed[key[0]][0], all_points_indexed[key[1]][0]],
                     [all_points_indexed[key[0]][1], all_points_indexed[key[1]][1]], 'b-', alpha=value)

            # clip value to 1
            # do the same but add an arrow at the end

            # plt.arrow(all_points_indexed[key[0]][0],
            #           all_points_indexed[key[0]][1],
            #           all_points_indexed[key[1]][0] - all_points_indexed[key[0]][0],
            #           all_points_indexed[key[1]][1] - all_points_indexed[key[0]][1],
            #           head_width=0.05, head_length=0.02,
            #           fc='b', ec='b', alpha=value)

    lab_so = False
    lab_si = False
    for mass, terminal in zip(masses, terminals):
        # terminal is a list with x,y coordinates, mass is the intensity
        color = 'ro' if mass < 0 else 'bo'
        lab = None
        if mass < 0 and not lab_so:
            lab = "Source"
            lab_so = True
        if mass > 0 and not lab_si:
            lab = "Sink"
            lab_si = True

        plt.plot(terminal[0], terminal[1], color, alpha=1, label=lab)

    print(f"Cost is:  {cost}")
    print(f"Flat_length: {flat_length}")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    db = Database.populate_from_folder("../../../run_logs/runtime_alpha")

    query = Query()

    query.add_filter(C("instance_arguments//alpha") == 0.5)

    exps = query.apply(db)

    visualize(exps[0])
    # visualize(exps[1])
    # visualize(exps[2])
    # visualize(exps[3])
    # visualize(exps[4])
    # visualize(exps[5])
