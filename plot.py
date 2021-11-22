import numpy as np
import pandas as pd
import seaborn as sns
import os
from run_real_benchmark import w_candidate

df = pd.DataFrame()
w = list(w_candidate)

for root, _, file_names in os.walk("focus-final-out"):
    for file_name in file_names:
        try:
            value = pd.read_csv(os.path.join(root, file_name), sep="\t", header=None).transpose()        
            df = df.append(value, ignore_index=True)
        except Exception:
            pass
    df = df.rename(
        index={i: file_names[i][:-4] for i in df.index},
        columns={i: w[i] for i in df.columns}
    )

    df = df.transpose()

# plot line
fig = sns.lineplot(data=df)
fig_name = "lineplot.png"
lineplt = fig.get_figure()
lineplt.savefig(fig_name, dpi=400)
