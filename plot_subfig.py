import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.rand(10, 2), columns=["x", "y"])
fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharex=True)
sns.lineplot(data=df, x="x", y="y", marker="o", markers=True, dashes=True, ax=axes[0])
plt.savefig("test.png")
plt.show()
print(df)


