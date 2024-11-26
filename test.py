import json
import os

from tqdm import tqdm

base_path = "./rankadaptor/results/for_dataset/"
files = os.listdir(base_path)
results = []

for file in tqdm(files):
    json_file = json.load(open(base_path + file, "r+"))

    performance = {}
    for key in json_file["results"].keys():
        result = json_file["results"][key]
        acc = result["acc"]
        performance[key] = acc

        if "acc_norm" in result.keys():
            acc_norm = result["acc_norm"]
            performance[key] = acc_norm if acc_norm > acc else acc

    item = {
        "name": file[:file.index("0.50") + 4],
        "rank_list": json_file["config"]["rank_list"],
        "pruning_rate_list": [
            0,
            0,
            0,
            0,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0.635,
            0,
            0
        ],
        "performance": performance
    }

    results.append(item)

print(results)

with open("dataset/raw_data/data_20241029.json", "w+") as file:
    file.write(json.dumps(results))
