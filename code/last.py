import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


output = pd.read_pickle("conf.pkl")
output = output.round(3)
print("ou: ", output)
plt.figure(figsize=(25,25))
sn.heatmap(output, annot=True)
plt.savefig('output.png')
