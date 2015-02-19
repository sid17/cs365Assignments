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
	print '###########'
	print x_min
	print x_max
	print '###########'
	

	plt.figure()
	ax = plt.subplot(111)
	print np.min(X, 0)*0.8
	ax.set_xlim([-13500,-10800])
	ax.set_ylim([17000,23000])
	# print int(np.max(X, 0)*1.2)
	# ax.set_xlim([,])
	# ax.set_ylim([int((np.min(X, 1)*0.8)[1]),int(np.max(X, 1)*1.2[1])])
	# ax.set_xlim([int(np.min(X, 0)*0.8[0]),int(np.max(X, 0)*1.2[0])])
	# ax.set_ylim([-3500,-4000])
	# ax.set_xlim([44000,45000])
	
	# ax.set_ylim([-0.0155,-0.0153])
	# ax.set_xlim([-0.01409,-0.01402])
	# -3500 -6000
	# -37000 -42000
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



with open('image_data.pkl', 'rb') as input:
	immatrix = pickle.load(input)

data=immatrix

print (sum(data.var(axis=0)))

mean = data.mean(axis=0)
data = (data - mean)

t0 = time()
print data.shape
mfold=manifold.Isomap(n_neighbors, lower_dimension)
Y = mfold.fit_transform(data)


kpca = KernelPCA(kernel="rbf", n_components=lower_dimension)
X_kpca = kpca.fit_transform(data)
   
# Y=X_kpca


nbrs = NearestNeighbors(n_neighbors=6).fit(Y)
distances_10, indices_10 = nbrs.kneighbors(Y[33,:])
distances_25, indices_25 = nbrs.kneighbors(Y[134,:])


with open('dictionary.pkl', 'rb') as input:
	dictionary = pickle.load(input)

# plt.figure()
# ax = plt.subplot(111)

# for i in range(0,6):
# 	print dictionary[indices_25[:,i][0]]
# 	img = cv2.imread(dictionary[indices_25[:,i][0]],0)
# 	cv2.imwrite("nearest_25_image"+str(i)+".jpg", img )
# 	plt.plot(Y[indices_25[:,i][0],:][0],Y[indices_25[:,i][0],:][1], ".r")



# for i in range(0,6):
# 	print dictionary[indices_10[:,i][0]]
# 	img = cv2.imread(dictionary[indices_10[:,i][0]],0)
# 	cv2.imwrite("nearest_10_image"+str(i)+".jpg", img )
	

# print Y[indices_25[:,:6][0],:]

# print dictionary[25]

# plot_embedding(Y[indices_25[:,:6][0],:],dictionary,0,indices_25[:,:6][0],6,'NearestNeighbors(Image=25')

# print Y[indices_25[:,:6][0],:]
plot_embedding(Y,dictionary,40000,range(143),143,'Lower Dimensional Embedding (Isomap, d=2)')

# plt.show()
plt.plot(Y[indices_10[:,0][0],:][0],Y[indices_10[:,0][0],:][1], 'ro')

for i in range (1,6):
	plt.plot(Y[indices_10[:,i][0],:][0],Y[indices_10[:,i][0],:][1], 'go')


plt.savefig('iso_embed.png', bbox_inches='tight')
# # plot_embedding(Y,dictionary,4e-3,range(143),'Plot me')


print Y[indices_10[:,:6][0],:]

# print Y
# # print dictionary
# # plot_embedding(Y, dictionary,4e-3,'Plot me')

# plt.show()
# # print indices
# t1 = time()
# print("Isomap: %.2g sec" % (t1 - t0))

# print (sum(Y.var(axis=0)))