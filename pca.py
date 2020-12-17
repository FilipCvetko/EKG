import time
import csv
import pandas as pd
import h5py
from sklearn.decomposition import IncrementalPCA, PCA
import numpy as np

file = h5py.File("./fourier_data/fourier_data.h5", "r")
data = file["data"] # Ni še naložen v memory.

num_rows = data.shape[0]
chunk_size = 1000

pca = PCA(n_components=50)
pca.fit(data)
pca_data = pca.transform(data)

# pca = IncrementalPCA(n_components=50, batch_size=16)

# for i in range(22):
#     print(i)
#     if i == 21:
#         pca.partial_fit(data[i*chunk_size : 21836])
#         continue
#     pca.partial_fit(data[i*chunk_size : (i+1)*chunk_size])

# pca_data = pca.transform(file["data"])
print(len(pca_data), len(pca_data[3]))
h5 = h5py.File("./pca_data/pca.h5", "w")
h5.create_dataset('data', data=pca_data, shape=(21837,50), dtype=np.float32)
h5.close()
