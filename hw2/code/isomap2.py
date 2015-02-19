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
from sklearn.decomposition import PCA, KernelPCA


def isomap_implement(n_neighbors,lower_dimension,no_of_samples,X):
    mfold=manifold.Isomap(n_neighbors, lower_dimension)
    Y = mfold.fit_transform(X)
    geodesic=mfold.dist_matrix_
    manifold_distance= euclidean_distances(Y,Y)
    A=reshape(geodesic,(no_of_samples**2))
    D=reshape(manifold_distance,(no_of_samples**2))
    r2 = 1-corrcoef(A,D)**2; 
    return r2[1][0],Y

def kpca_implement(lower_dimension,X):
    kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, n_components=lower_dimension)
    X_kpca = kpca.fit_transform(X)
    return sum(X_kpca.var(axis=0)),X_kpca 

with open('image_data.pkl', 'rb') as input:
    immatrix = pickle.load(input)

X=immatrix
R=np.zeros((5,10));
j=0;
output_polyfit_small=""
no_of_samples=143
n_neighbors=5
X_plot=[]
Y_plot=[]

isomap=0

with open('dictionary.pkl', 'rb') as input:
    dictionary = pickle.load(input)

for lower_dimension in range(1,11):
    # R[j][lower_dimension]=isomap_implement(n_neighbors,lower_dimension,no_of_samples,X);
    if isomap==1:
        residual,Y=isomap_implement(n_neighbors,lower_dimension,no_of_samples,X)
        # print n_neighbors,lower_dimension,residual
        X_plot.append(lower_dimension)
        Y_plot.append(residual)

        nbrs = NearestNeighbors(n_neighbors=6).fit(Y)
        distances_10, indices_10 = nbrs.kneighbors(Y[10,:])
        distances_25, indices_25 = nbrs.kneighbors(Y[25,:])
        
    else:
        residual,Y=kpca_implement(lower_dimension,X)
        sum(Y.var(axis=0))
        # print n_neighbors,lower_dimension,residual
        X_plot.append(lower_dimension)
        Y_plot.append(residual/sum(X.var(axis=0)))
        nbrs = NearestNeighbors(n_neighbors=6).fit(Y)
        distances_10, indices_10 = nbrs.kneighbors(Y[10,:])
        distances_25, indices_25 = nbrs.kneighbors(Y[25,:])
if isomap==1:
    plt.plot(X_plot,Y_plot,color='r', linewidth=2.0)
    plt.ylabel('Residual Variance')
    plt.xlabel('Dimensionality of the Embedding')
    plt.suptitle('Graph for Residual Variance vs Dimensionality of the Embedding(Isomap)')
    plt.savefig('isomap_implement.png', bbox_inches='tight')
else :
    plt.plot(X_plot,Y_plot,color='r', linewidth=2.0)
    plt.ylabel('Percentage of Variance')
    plt.xlabel('Dimensionality of the Embedding')
    plt.suptitle('Graph for Residual Variance vs Dimensionality of the Embedding(Kernel PCA)')
    plt.savefig('kpca_implement.png', bbox_inches='tight')