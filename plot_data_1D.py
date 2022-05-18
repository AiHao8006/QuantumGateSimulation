import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("Data_saved.xlsx")

df = df.sort_values(by='t_f')

t_f = df['t_f']
Fid = df['Fidelity']

plt.plot(t_f, Fid)
plt.show()

