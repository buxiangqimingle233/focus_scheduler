import seaborn as sns
import pandas as pd
import numpy as np


df = pd.read_csv("result.csv")

fig = sns.lineplot(data=df, x="performance", y="factor", marker="o")
fig_name = "lineplot.png"
lineplt = fig.get_figure()
lineplt.savefig(fig_name, dpi=400)