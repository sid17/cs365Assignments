
###################Usage ###########################

# Y is the data vector to be visualised
# dictionary is the mapping : dictionary[1]='/home/siddhantmanocha...../image1.jpg' of image locations
#4e-3 is the distance such that if two points are at this distance apart then image will be plotted else 
#image will be plotted for one of them
#range(143) list [1,2,3,....143] which shows that the image Y[1] corresponds to the dictionary[1] image 
#else image 1 Y[1] will correspond to dictionary[listelement[i]] 
# 143 is the number of points 
#NearestNeighbors(Image=25) : this is the title

# :::plot_embedding(Y,dictionary,4e-3,range(143),143,'NearestNeighbors(Image=25)')

##############################################


##########################################
# Set the x lim and ylim in function plot_embedding for zoom in feature
#############################################


import os
import numpy as np
import cv2
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold
import pickle
from sklearn.neighbors import NearestNeighbors
from matplotlib import offsetbox
from PIL import Image
from sklearn.decomposition import PCA, KernelPCA

n_neighbors = 5
lower_dimension=2


def plot_embedding(X, dictionary,epsilon,indices,n_sam,title='None'):
	x_min, x_max = np.min(X, 0)-np.min(X, 0)*0.2, np.max(X, 0)+np.max(X, 0)*0.2
	# X = (X - x_min) / (x_max - x_min)
	
	plt.figure()
	ax = plt.subplot(111)
	print np.min(X, 0)*0.8
	#################################SET THE LIMIT FOR ZOOM IN############################
	# ax.set_ylim([-3500,-4000])
	# ax.set_xlim([44000,45000])
	#################################SET THE LIMIT FOR ZOOM IN############################
	if hasattr(offsetbox, 'AnnotationBbox'):
		# only print thumbnails with matplotlib > 1.0
		shown_images = np.array([[1., 1]])  # just something big
		for i in range(n_sam):
			dist = np.sum((X[i] - shown_images) ** 2, 1)
			plt.plot(X[i][0], X[i][1], 'bo')
			print X[i][0]
			print X[i][1]
			if np.min(dist) < epsilon:
				# don't show points that are too close
				continue

			shown_images = np.r_[shown_images, [X[i]]]
			img=Image.open(dictionary[indices[i]])
			imagebox = offsetbox.AnnotationBbox(
				offsetbox.OffsetImage(img, cmap=plt.cm.gray_r,zoom=0.3),
				X[i],xybox=(-30, 50),xycoords='data',boxcoords="offset points",arrowprops=dict(arrowstyle="->"),pad=0)
			ax.add_artist(imagebox)
	plt.xticks([]), plt.yticks([])
	plt.axis()
	if title is not None:
		plt.title(title)



with open('image_data.pkl', 'rb') as input:
	immatrix = pickle.load(input)

data=immatrix
mean = data.mean(axis=0)
data = (data - mean)
t0 = time()
mfold=manifold.Isomap(n_neighbors, lower_dimension)
Y = mfold.fit_transform(data)


kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, n_components=lower_dimension)
X_kpca = kpca.fit_transform(data)
   
nbrs = NearestNeighbors(n_neighbors=6).fit(Y)
distances_10, indices_10 = nbrs.kneighbors(Y[10,:])
distances_25, indices_25 = nbrs.kneighbors(Y[25,:])

with open('dictionary.pkl', 'rb') as input:
	dictionary = pickle.load(input)

plot_embedding(Y,dictionary,4e-3,range(143),143,'NearestNeighbors(Image=25)')


# Uncomment ot plot the nearest neighbour of 10 in different color , green in this case
# for i in range (1,6):
# 	plt.plot(Y[indices_10[:,i][0],:][0],Y[indices_10[:,i][0],:][1], 'go')


plt.savefig('iso_embed.png', bbox_inches='tight')

