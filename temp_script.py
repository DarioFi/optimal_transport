import json
import os

"""we need to read all the files from "runs/" which are json with lists of dictionaries, modify the dict and rewrite everything"""

folder = "old_runs/"

for file in os.listdir(folder):
    if file.endswith(".json"):
        with open(os.path.join(folder, file), 'r') as f:
            data = json.load(f)
            assert type(data) == list
            for run in data:
                if "alpha" in run["formulation_arguments"]:

                    alpha = run["formulation_arguments"]["alpha"]
                    del run["formulation_arguments"]["alpha"]

                    run["instance_arguments"]["alpha"] = alpha
                    run["instance"]["alpha"] = alpha

        # write the modified dict back
        with open(os.path.join(folder, file), 'w') as f:
            json.dump(data, f, indent=4)
