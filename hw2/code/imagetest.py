import cv2
import pickle
import numpy as np
image_height=115
image_width=144
with open('image_data.pkl', 'rb') as input:
    immatrix = pickle.load(input)

with open('dictionary.pkl', 'rb') as input:
    dictionary = pickle.load(input)

reconstruction_ex=immatrix[25,:]
r_channel=reconstruction_ex[:image_height*image_width]
g_channel=reconstruction_ex[image_height*image_width:image_height*image_width*2]
b_channel=reconstruction_ex[image_height*image_width*2:image_height*image_width*3]

A_r= np.transpose(((r_channel).reshape((image_height,image_width))))
A_g= np.transpose(((g_channel).reshape((image_height,image_width))))
A_b= np.transpose(((b_channel).reshape((image_height,image_width))))

rgb = np.dstack((A_r,A_g,A_b))

cv2.imwrite("test.jpg", rgb )

print dictionary[25]