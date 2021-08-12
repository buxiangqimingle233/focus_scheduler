import os
import sys
import re

# checked_model = "resnext50_32x4d"
checked_model = "mnasnet"
checked_model = "wide_resnet50_2"
# checked_model = "resnet50"

root = os.path.dirname(os.path.abspath(__file__))
prob_dir = os.path.join(root, "db/", checked_model)

res = "["
for father, _, files in os.walk(prob_dir):
    for file in files:
        with open(os.path.join(father, file)) as f:
            content = f.read()
            if content.find("M") < 0:
                # print("layer: {} does not have dimension M!".format(file))
                res += re.search(r"([0-9]+)", file).group(1) + ", "

res += "]"
print(res)
