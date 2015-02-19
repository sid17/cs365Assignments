import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
# xTrue = np.linspace(0, 1000, N)
# yTrue = 3 * xTrue
# xData = xTrue + np.random.normal(0, 100, N)
# yData = yTrue + np.random.normal(0, 100, N)
# xData = np.reshape(xData, (N, 1))
# yData = np.reshape(yData, (N, 1))
# data = np.hstack((xData, yData))

path='/home/satvikg/sid/cs365/hw2'
paths = [os.path.join(path,fn) for fn in next(os.walk(path))[2]]
# print len(paths)
im_width=720
im_height=576
#immatrix=np.zeros((len(paths), im_width*im_height))
#imm=np.zeros((1,im_width*im_height))

#for var in range(0,len(paths)): #len(paths)
#	img = cv2.imread(paths[var],0)
#	img_flat=img.flatten()
#	img_flat=np.transpose(img_flat)
#	immatrix[var:]=img_flat
	
#data=immatrix

#with open('company_data.pkl', 'wb') as output:
#    pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

with open('company_data.pkl', 'rb') as input:
    cfd = pickle.load(input)

data=cfd

mean = data.mean(axis=0)

print mean.shape
data = (data - mean)

print sum(data.std(axis=0) )

# print data.shape

eigenvectors, eigenvalues, V = np.linalg.svd(data, full_matrices=False)

print eigenvectors.shape

with open('company_eigenval.pkl', 'wb') as output:
    pickle.dump(eigenvalues, output, pickle.HIGHEST_PROTOCOL)
#eigenvectors_A=np.dot(data.T, eigenvectors)

#print(eigenvectors_A.shape)
# print eigenvectors_A.shape

# print (eigenvectors_A[:,1]).shape
# # B=
#A=np.reshape(eigenvectors_A[:,1], (720,576))

#with open('company_eigen.pkl', 'wb') as output:
#    pickle.dump(eigenvectors_A, output, pickle.HIGHEST_PROTOCOL)

with open('company_eigen.pkl', 'rb') as input:
    load_eigen = pickle.load(input)

A=np.reshape(load_eigen[:,1], (720,576))
print (load_eigen.shape)
cv2.imwrite("face.jpg", A)

projected_data=np.dot(data,load_eigen)

print (projected_data.shape)
#cv2.imshow('image',A)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# # mu = data.mean(axis=0)

# # data = data - mu

# # # data = (data - mu)/data.std(axis=0)  # Uncomment this reproduces mlab.PCA results
# # eigenvectors, eigenvalues, V = np.linalg.svd(
# #     data.T, full_matrices=False)

# # print eigenvectors.shape

# # projected_data = np.dot(data, eigenvectors)

# # print projected_data.shape

# # sigma = projected_data.std(axis=0).mean()

# # # print eigenvalues
# # # print (sigma)
# # def annotate(ax, name, start, end):
# #     arrow = ax.annotate(name,
# #                         xy=end, xycoords='data',
# #                         xytext=start, textcoords='data',
# #                         arrowprops=dict(facecolor='red', width=2.0))
# #     return arrow

# # fig, ax = plt.subplots()
# # ax.scatter(xData, yData)
# # ax.set_aspect('equal')
# # for axis in eigenvectors:
# #     annotate(ax, '', mu, mu + sigma * axis)
# # plt.show()
