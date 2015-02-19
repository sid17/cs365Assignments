from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from PIL import Image
import pickle
import os
import numpy as np
path='/home/siddhantmanocha/assignments/cs365/back/cs365/hw2_updated'
paths = [os.path.join(path,fn) for fn in next(os.walk(path))[2]]
j=1

animals = {}

for i in range(len(paths)):
	animals[i]=paths[i]
print animals[1]
im_size=np.array(Image.open(paths[0]).convert('L'))
print (np.array(Image.open(paths[0]))[:,:,0].shape)
immatrix_red=np.array([np.array(Image.open(paths[i]))[:,:,0].flatten() for i in range(len(paths))],'f')
immatrix_green=np.array([np.array(Image.open(paths[i]))[:,:,1].flatten() for i in range(len(paths))],'f')
immatrix_blue=np.array([np.array(Image.open(paths[i]))[:,:,2].flatten() for i in range(len(paths))],'f')
immatrix=np.concatenate((immatrix_blue,immatrix_green,immatrix_red),axis=1)

import cv2
image_width=144
image_height=115
reconstruction_ex=immatrix[25,:]
r_channel=reconstruction_ex[:image_height*image_width]
g_channel=reconstruction_ex[image_height*image_width:image_height*image_width*2]
b_channel=reconstruction_ex[image_height*image_width*2:image_height*image_width*3]

# A_r= np.transpose(((r_channel).reshape((image_height,image_width))))
# A_g= np.transpose(((g_channel).reshape((image_height,image_width))))
# A_b= np.transpose(((b_channel).reshape((image_height,image_width))))

# rgb = np.dstack((np.transpose(A_r),np.transpose(A_g),np.transpose(A_b)))

# # ((immatrix_red[25,:]).reshape((image_height,image_width)))
# cv2.imwrite("test.jpg", rgb )



with open('image_data.pkl', 'wb') as output:
   pickle.dump(immatrix, output, pickle.HIGHEST_PROTOCOL)

with open('dictionary.pkl', 'wb') as output:
   pickle.dump(animals, output, pickle.HIGHEST_PROTOCOL)