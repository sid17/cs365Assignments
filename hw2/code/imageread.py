import os
import numpy as np
import cv2
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from PIL import Image
import pickle
from matplotlib import offsetbox
from sklearn.neighbors import NearestNeighbors

def plot_embedding(X, dictionary,epsilon,indices,n_sam,title='None'):
	x_min, x_max = np.min(X, 0)-np.min(X, 0)*0.2, np.max(X, 0)+np.max(X, 0)*0.2
	X = (X - x_min) / (x_max - x_min)
	
	plt.figure()
	ax = plt.subplot(111)

	# plt.axis([0, 6, 0, 20])
	if hasattr(offsetbox, 'AnnotationBbox'):
		# only print thumbnails with matplotlib > 1.0
		shown_images = np.array([[1., 1]])  # just something big
		for i in range(n_sam):
			dist = np.sum((X[i] - shown_images) ** 2, 1)
			plt.plot(X[i][0], X[i][1], 'bo')
			if np.min(dist) < epsilon:
				# don't show points that are too close
				continue

			shown_images = np.r_[shown_images, [X[i]]]
			# Image.open("/home/siddhantmanocha/assignments/cs365/back/cs365/hw2_updated/scene00089.jpg").resize((30, 30), Image.BILINEAR)
			img=Image.open(dictionary[indices[i]])
			imagebox = offsetbox.AnnotationBbox(
				offsetbox.OffsetImage(img, cmap=plt.cm.gray_r,zoom=0.3),
				X[i],xybox=(-30, 50),xycoords='data',boxcoords="offset points",arrowprops=dict(arrowstyle="->"),pad=0)
			ax.add_artist(imagebox)
	plt.xticks([]), plt.yticks([])
	plt.axis()
	if title is not None:
		plt.title(title)


def implement_pca(lower_dimension,image_height,image_width):
	
	with open('image_data.pkl', 'rb') as input:
	    immatrix = pickle.load(input)

	data=immatrix
	mean = data.mean(axis=0)

	original_variance=sum(data.var(axis=0))

	data = (data - mean)

	eigenvectors, eigenvalues, V = np.linalg.svd(data, full_matrices=False)

	projected_data=np.dot(data, V[:lower_dimension,:].T)


	# plot_embedding(projected_data,dictionary,4e-3,range(143),143,'Lower Dimensional Embedding(Dimension=2)(PCA)')
	# plt.savefig('pca_embed.png', bbox_inches='tight')


	projected_variance=sum(projected_data.var(axis=0))

	reconstruction=np.dot(projected_data,V[:lower_dimension,:])

	reconstruction_ex=reconstruction[134,:]+mean

	print dictionary


	nbrs = NearestNeighbors(n_neighbors=11).fit(projected_data)

	distances_25, indices_25 = nbrs.kneighbors(projected_data[134,:])

	print indices_25

	recons=np.zeros(image_height*image_width*3)

	dist_norm=0
	for i in range(1,11):
		dist_norm=dist_norm+1/distances_25[:,i][0]

	for i in range(1,11):
		print distances_25[:,i][0]/dist_norm
		recons=recons+(data[indices_25[:,i][0],:]/distances_25[:,i][0])/dist_norm
		# recons=recons+reconstruction[indices_25[:,i][0],:]*distances_25[:,i][0]/dist_norm

		print dictionary[indices_25[:,i][0]]
		# img = cv2.imread(dictionary[indices_10[:,i][0]],0)
		# cv2.imwrite("nearest_10_image"+str(i)+".jpg", img )
	recons=recons+mean
	print reconstruction_ex.shape
	print recons.shape

	reconstruction_ex=recons

	r_channel=reconstruction_ex[:image_height*image_width]
	g_channel=reconstruction_ex[image_height*image_width:image_height*image_width*2]
	b_channel=reconstruction_ex[image_height*image_width*2:image_height*image_width*3]

	A_r= np.transpose(((r_channel).reshape((image_height,image_width))))
	A_g= np.transpose(((g_channel).reshape((image_height,image_width))))
	A_b= np.transpose(((b_channel).reshape((image_height,image_width))))

	rgb = np.dstack((np.transpose(A_r),np.transpose(A_g),np.transpose(A_b)))


	cv2.imwrite("image"+str(lower_dimension)+"_o.jpg", rgb )

	return projected_variance/original_variance

X=[]
Y=[]
with open('dictionary.pkl', 'rb') as input:
	dictionary = pickle.load(input)

for lower_dimension in [2,10,30,80]:
	ratio_variance=implement_pca(lower_dimension,image_height=115,image_width=144)
	X.append(lower_dimension)
	Y.append(ratio_variance)
	

plt.plot(X,Y,color='r', linewidth=2.0)
plt.ylabel('Percentage of Variance Preserved')
plt.xlabel('Dimensionality of the Embedding')
plt.suptitle('Graph for Percentage Variance Preserved vs Dimensionality of the Embedding(PCA)')
plt.savefig('pca.png', bbox_inches='tight')
