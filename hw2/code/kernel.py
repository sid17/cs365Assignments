import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA, KernelPCA

with open('image_data.pkl', 'rb') as input:
    immatrix = pickle.load(input)

lower_dimension=2

data=immatrix

mean = data.mean(axis=0)

print (sum(data.var(axis=0)))

data = (data - mean)

kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10,n_components=2)
X_kpca = kpca.fit_transform(data)

print (sum(X_kpca.var(axis=0)))

