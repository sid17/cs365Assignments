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
import cv2

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
    kpca = KernelPCA(kernel="rbf", n_components=lower_dimension)
    X_kpca = kpca.fit_transform(X)
    return sum(X_kpca.var(axis=0)),X_kpca 

def pca_implement(lower_dimension,X,image_height=115,image_width=144):
    pca = PCA(n_components=lower_dimension)
    Y=pca.fit_transform(X)
    X_r=pca.inverse_transform(Y)
    reconstruction_ex=X_r[25,:]
    r_channel=reconstruction_ex[:image_height*image_width]
    g_channel=reconstruction_ex[image_height*image_width:image_height*image_width*2]
    b_channel=reconstruction_ex[image_height*image_width*2:image_height*image_width*3]

    A_r= np.transpose(((r_channel).reshape((image_height,image_width))))
    A_g= np.transpose(((g_channel).reshape((image_height,image_width))))
    A_b= np.transpose(((b_channel).reshape((image_height,image_width))))

    rgb = np.dstack((A_r,A_g,A_b))


    cv2.imwrite("test.jpg", rgb )
    # cv2.imwrite('',)

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

isomap=1

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
        distances_10, indices_10 = nbrs.kneighbors(Y[33,:])
        distances_25, indices_25 = nbrs.kneighbors(Y[134,:])
        print 
        print '###Dimensionality='+str(lower_dimension)
        print 
        print '####NearestNeighbors(Image25)'
        print
        for i in range(0,6,3):
            # print dictionary[indices_25[:,i][0]]
            print '| <img src="/images/hw2_updated/'+dictionary[indices_25[:,i][0]].split('/')[-1]+'"  style="width:100%">  |<img src="/images/hw2_updated/'+dictionary[indices_25[:,i+1][0]].split('/')[-1]+'"  style="width:100%"> |<img src="/images/hw2_updated/'+dictionary[indices_25[:,i+2][0]].split('/')[-1]+'"  style="width:100%"> |'
            print '|:-----------------:|:-----------:|:-----------:|'
            if (i==0):
                print '| Original Image              |Neighbour'+str(i+1)+' |Neighbour'+str(i+2)+' |'
            else:
                print '| Neighbour'+str(i)+'             |Neighbour'+str(i+1)+' |Neighbour'+str(i+2)+' |'
            print 
            # img = cv2.imread(dictionary[indices_25[:,i][0]],0)
            # cv2.imwrite("nearest_25_image"+str(i)+".jpg", img )
            # plt.plot(Y[indices_25[:,i][0],:][0],Y[indices_25[:,i][0],:][1], ".r")
        print    
        print '####NearestNeighbors(Image10)'
        print
        for i in range(0,6,3):
            # print dictionary[indices_25[:,i][0]]
            print '| <img src="/images/hw2_updated/'+dictionary[indices_10[:,i][0]].split('/')[-1]+'"  style="width:100%">  |<img src="/images/hw2_updated/'+dictionary[indices_10[:,i+1][0]].split('/')[-1]+'"  style="width:100%"> |<img src="/images/hw2_updated/'+dictionary[indices_10[:,i+2][0]].split('/')[-1]+'"  style="width:100%"> |'
            print '|:-----------------:|:-----------:|:-----------:|'
            if (i==0):
                print '| Original Image              |Neighbour'+str(i+1)+' |Neighbour'+str(i+2)+' |'
            else:
                print '| Neighbour'+str(i)+'             |Neighbour'+str(i+1)+' |Neighbour'+str(i+2)+' |'
            print 
    elif isomap==2:
        residual,Y=kpca_implement(lower_dimension,X)
        sum(Y.var(axis=0))
        # print n_neighbors,lower_dimension,residual
        X_plot.append(lower_dimension)
        Y_plot.append(residual/sum(X.var(axis=0)))
        # nbrs = NearestNeighbors(n_neighbors=6).fit(Y)

        print Y.shape
        print Y[33,:]
        print Y[134,:]
        nbrs = NearestNeighbors(n_neighbors=6).fit(Y)
        distances_10, indices_10 = nbrs.kneighbors(Y[33,:])
        distances_25, indices_25 = nbrs.kneighbors(Y[134,:])
        print indices_25
        print 
        print '###Dimensionality='+str(lower_dimension)
        print 
        print '####NearestNeighbors(Image25)'
        print
        for i in range(0,6,3):
            # print dictionary[indices_25[:,i][0]]
            print '| <img src="/images/hw2_updated/'+dictionary[indices_25[:,i][0]].split('/')[-1]+'"  style="width:100%">  |<img src="/images/hw2_updated/'+dictionary[indices_25[:,i+1][0]].split('/')[-1]+'"  style="width:100%"> |<img src="/images/hw2_updated/'+dictionary[indices_25[:,i+2][0]].split('/')[-1]+'"  style="width:100%"> |'
            print '|:-----------------:|:-----------:|:-----------:|'
            if (i==0):
                print '| Original Image              |Neighbour'+str(i+1)+' |Neighbour'+str(i+2)+' |'
            else:
                print '| Neighbour'+str(i)+'             |Neighbour'+str(i+1)+' |Neighbour'+str(i+2)+' |'
            print 
            # img = cv2.imread(dictionary[indices_25[:,i][0]],0)
            # cv2.imwrite("nearest_25_image"+str(i)+".jpg", img )
            # plt.plot(Y[indices_25[:,i][0],:][0],Y[indices_25[:,i][0],:][1], ".r")
        print    
        print '####NearestNeighbors(Image10)'
        print
        for i in range(0,6,3):
            # print dictionary[indices_25[:,i][0]]
            print '| <img src="/images/hw2_updated/'+dictionary[indices_10[:,i][0]].split('/')[-1]+'"  style="width:100%">  |<img src="/images/hw2_updated/'+dictionary[indices_10[:,i+1][0]].split('/')[-1]+'"  style="width:100%"> |<img src="/images/hw2_updated/'+dictionary[indices_10[:,i+2][0]].split('/')[-1]+'"  style="width:100%"> |'
            print '|:-----------------:|:-----------:|:-----------:|'
            if (i==0):
                print '| Original Image              |Neighbour'+str(i+1)+' |Neighbour'+str(i+2)+' |'
            else:
                print '| Neighbour'+str(i)+'             |Neighbour'+str(i+1)+' |Neighbour'+str(i+2)+' |'
            print 
    else:
        print lower_dimension
        pca_implement(lower_dimension,X)
        # img = cv2.imread(dictionary[indices_25[:,i][0]],0)
        # cv2.imwrite("nearest_25_image"+str(i)+".jpg", img )
        # plt.plot(Y[indices_25[:,i][0],:][0],Y[indices_25[:,i][0],:][1], ".r")
        # img = cv2.imread(dictionary[indices_10[:,i][0]],0)
        # cv2.imwrite("nearest_10_image"+str(i)+".jpg", img )
    # plt.plot([1,2,3,4,5,6,7,8,9,10], R[j],'ro',color = 'green',linestyle='dashed')
    # plt.ylabel('Residual Variance with ' + str(n_neighbors) + 'nearest neighbours')
    # plt.xlabel('Different dimensions of reduction')
    # plt.title('Residual Variance with ' + str(n_neighbors) + 'nearest neighbours vs different dimensions of reduction')
    # plt.savefig('Nearest neighbour '+str(n_neighbors)+'.png')
    # plt.close()

    output_polyfit_small += "<p>Following figure shows "+'Residual Variance with ' + str(n_neighbors) + 'nearest neighbours vs different dimensions of reduction'+".<br><image src=\""+'Nearest neighbour '+str(n_neighbors)+'.png'+"\"><br>"
    j+=1;

    # for i in range(1,10):
	   #      plt.plot([3,5,7,10,15], R[:,i],'ro',color = 'green',linestyle='dashed')
    #         plt.ylabel('Residual Variance with ' + str(i) + 'dimensions of reduction')
    #         plt.xlabel('K nearest neighbour')
    #         plt.title('Residual Variance with ' + str(i) + 'dimension of reduction vs k nearest neighbour')
    #         plt.savefig('Dimension of reduction '+str(i)+'.png')
    #         plt.close()
    #         output_polyfit_small += "<p>Following figure shows "+'Residual Variance with ' + str(i) + 'dimension of reduction vs k nearest neighbour'+".<br><image src=\""+'Dimension of reduction '+str(i)+'.png'+"\"><br>"
    # print output_polyfit_small;
if isomap==1:
    print X_plot
    print Y_plot
    plt.plot(X_plot,Y_plot,color='r', linewidth=2.0)
    plt.ylabel('Residual Variance')
    plt.xlabel('Dimensionality of the Embedding')
    plt.suptitle('Graph for Residual Variance vs Dimensionality of the Embedding(Isomap)')
    plt.savefig('isomap_implement.png', bbox_inches='tight')
else :
    plt.plot(X_plot,Y_plot,color='r', linewidth=2.0)
    plt.ylabel('Percentage of Variance')
    plt.xlabel('Dimensionality of the Embedding')
    plt.suptitle('Graph for Residual Variance vs Dimensionality of the Embedding(Kernel PCA)(sigmoid kernel)')
    plt.savefig('sigmoid.png', bbox_inches='tight')