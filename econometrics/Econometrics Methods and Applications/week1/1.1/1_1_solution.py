import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Dataset1.csv', sep='	',header=1, index_col =0)

data.plot(kind='bar')
plt.ylabel('Frequency')
plt.xlabel('Words')
plt.title('Title')

plt.show()
