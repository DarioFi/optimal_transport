import subprocess


def euclidean_ST(point_list):
    """
    Runs the commands 'rand_points', 'efst', and 'bb -f' sequentially,
    piping the output of each to the next, and captures the final stdout.

    Args:
        point_list: A list of lists, where each inner list contains [x, y] coordinates.
                    Example: [[1.0, 4.2], [2.5, 5.7], [3.1, 6.9]]

    Returns:
        A string containing the stdout of the 'bb -f' command, or None if an error occurs.
    """

    num_points = len(point_list)
    if not point_list:
        print("Error: The point list is empty.")
        return None

    rand_points_input = ""
    for point in point_list:
        if len(point) == 2:
            rand_points_input += f"{point[0]} {point[1]}\n"
        else:
            print("Error: Each inner list in point_list must contain exactly two coordinates (x, y).")
            return None

    try:
        # Run efst and pipe its output to bb -f
        efst_process = subprocess.Popen(
            ["efst"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True
        )
        bb_stdout, bb_stderr = efst_process.communicate(input=rand_points_input)

        if efst_process.returncode != 0:
            print(f"Error running efst. Return code: {efst_process.returncode}")
            if bb_stderr:
                print(f"efst stderr:\n{bb_stderr}")
            return None

        # Run bb -f and capture its stdout
        bb_process = subprocess.Popen(
            ["bb", "-f"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        final_stdout, final_stderr = bb_process.communicate(input=bb_stdout)

        if bb_process.returncode != 0:
            print(f"Error running bb -f. Return code: {bb_process.returncode}")
            if final_stderr:
                print(f"bb -f stderr:\n{final_stderr}")
            return None

        return final_stdout

    except FileNotFoundError as e:
        print(f"Error: Command not found: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def process_output(output):
    out_lines = output.split('\n')

    result = iter(out_lines)

    def line():
        return next(result)

    METRICS = ['Rectilinear', 'Euclidean', 'Graph']
    FST_STATUS = ["never needed", "maybe needed", "always needed"]

    data = {
        "version": line(),
        "instance": line(),
        "metric": METRICS[int(line())],
        "number_of_terminals": int(line()),
        "optimal_cost": float(line().split()[0]),
        "scaling_factor": line(),
        "decimal_integrality": float(line().split()[0]),
        "machine": line(),
        "cpu_time": int(line()),
        "number_of_FSTs": int(line()),
        "terminals_coords": [],
        "terminal_is_not_a_steiner_point": [],
        "fsts": [],
    }

    for _ in range(data["number_of_terminals"]):
        x, y = line().split()[:2]
        data["terminals_coords"].append([
            float(x),
            float(y),
        ])

    data["terminal_is_not_a_steiner_point"] = [int(x) for x in line().split()]

    for _ in range(data["number_of_FSTs"]):
        fst_data = {
            "number_of_terminals": int(line()),
            "terminals_indices": [int(x) for x in line().split()],
            "length": float(line().split()[0]),
            "n_steiner": int(line()),
            "steiner_coords": [],
        }

        for _ in range(fst_data["n_steiner"]):
            x, y = line().split()[:2]
            fst_data["steiner_coords"].append([
                float(x),
                float(y),
            ])

        fst_data["n_edges"] = int(line())
        fst_data["edges"] = []
        for _ in range(fst_data["n_edges"]):
            origin, dest = line().split()
            fst_data["edges"].append([int(origin), int(dest)])

        fst_data["status"] = FST_STATUS[int(line())]
        fst_data["n_incompatible_fsts"] = int(line())
        if fst_data["n_incompatible_fsts"]:
            fst_data["incompatible_fsts"] = line().split()

        data["fsts"].append(fst_data)

    return data


def EST(terminals):
    o = euclidean_ST(terminals)
    return process_output(o)


def compute_cost(terminals):
    return process_output(euclidean_ST(terminals))['optimal_cost']
