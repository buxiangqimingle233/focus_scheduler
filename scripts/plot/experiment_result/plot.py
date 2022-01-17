import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mlperf = pd.read_csv("mlperf.dat", header=None, sep=" ")
resnets = pd.read_csv("resnets.dat", header=None, sep="\t")
bert = pd.read_csv("bert.dat", header=None, sep="\t")

mlperf.to_csv("mlperf.dat", index=False, sep=" ", header=0)
resnets.to_csv("resnets.dat", index=False, sep=" ", header=0)
bert.to_csv("bert.dat", index=False, sep=" ", header=0)

columns = ["METERO", "DO", "XYYX", "ROMM", "ADP"]

plt.figure(figsize=(10, 6))
fig, axes = plt.subplots(1, 3)
sns.lineplot(data=mlperf, ax=axes[0], marker="o")
sns.lineplot(data=resnets, ax=axes[1], marker="o")
sns.lineplot(data=bert, ax=axes[2], marker="o")
plt.savefig("fkfkf.png")