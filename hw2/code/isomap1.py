import numpy as NP
from scipy import linalg as LA
from PIL import Image
from pylab import imread,subplot,imshow,title,gray,figure,show,NullLocator
from numpy import mean,cov,cumsum,dot,linalg,size,flipud,corrcoef,reshape
from sklearn import decomposition
from sklearn import datasets
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets
import numpy
from PIL import Image
import pickle 
import os
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors

def isomap_implement(n_neighbors,lower_dimension,no_of_samples,X):
    mfold=manifold.Isomap(n_neighbors, lower_dimension)
    Y = mfold.fit_transform(X)
    geodesic=mfold.dist_matrix_
    manifold_distance= euclidean_distances(Y,Y)
    A=reshape(geodesic,(no_of_samples**2))
    D=reshape(manifold_distance,(no_of_samples**2))
    r2 = 1-corrcoef(A,D)**2; 
    return r2[1][0]

with open('image_data.pkl', 'rb') as input:
    immatrix = pickle.load(input)

X=immatrix
R=np.zeros((5,10));
j=0;
output_polyfit_small=""
no_of_samples=143
n_neighbors=5
X=[]
Y=[]
for lower_dimension in range(1,10):
    residual=isomap_implement(n_neighbors,lower_dimension,no_of_samples,X)
    print n_neighbors,lower_dimension,residual
    output_polyfit_small += "<p>Following figure shows "+'Residual Variance with ' + str(n_neighbors) + 'nearest neighbours vs different dimensions of reduction'+".<br><image src=\""+'Nearest neighbour '+str(n_neighbors)+'.png'+"\"><br>"
    X.append(lower_dimension)
    Y.append(residual)
plot(X,Y)