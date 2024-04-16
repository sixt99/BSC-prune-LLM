import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("models/outputs/output_a0.1_bs128.csv")

mat = np.array([x['eval_matthews'] for x in data])
idxs = np.argsort(mat)
weights = np.zeros([6,6])
for i in idxs[-10:]:
    print(data[i]['eval_matthews'])
    for i, j in data[i]['pairs']:
        weights[i,j] += 1
plt.imshow(weights)
plt.colorbar()
plt.show()


#matrix = np.zeros([6,6])
#weights = np.zeros([6,6])
#for row in data:
#    for i, j in row['pairs']:
#        matrix[i,j] += row['eval_matthews']
#        weights[i,j] += 1
#matrix/=weights
#plt.imshow(matrix)
#plt.colorbar()
#plt.show()