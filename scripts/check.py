import os
import sys
import re
import yaml

# checked_model = "resnext50_32x4d"
# checked_model = "unet"
# checked_model = "mnasnet"
# checked_model = "inception"
checked_model = "mobilenet_v3_large"
# checked_model = "wide_resnet50_2"
# checked_model = "resnet50"

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prob_dir = os.path.join(root, "database", checked_model)

res = []
for father, _, files in os.walk(prob_dir):
    for file in files:
        obj = yaml.load(open(os.path.join(father, file), "r"), Loader=yaml.FullLoader)
        if "M" not in obj["problem"]["instance"] or "M" not in obj["problem"]["shape"]["dimensions"]:
            obj["problem"]["instance"]["M"] = 1
            obj["problem"]["shape"]["dimensions"].append("M")
            res.append(int(re.search(r"([0-9]+)", file[::-1]).group(1)[::-1]))
            yaml.dump(obj, open(os.path.join(father, file), "w"))
        # with open(os.path.join(father, file)) as f:
        #     content = f.read()
        #     if content.find("M") < 0:
        #         print("layer: {} does not have dimension M!".format(file))
        #         res.append(int(re.search(r"([0-9]+)", file).group(1)))

res.sort()

print(res)
