import Image
import numpy as np
import os

def imgResize(im,i):
    width = 288
    height = 230
    im3 = im.resize((width, height), Image.BILINEAR) # linear interpolation in a 2x2 environment
    im3.save(i)
    
def main():

	path='/home/siddhantmanocha/assignments/cs365/back/cs365/images'
	paths = [os.path.join(path,fn) for fn in next(os.walk(path))[2]]
	for i in range(len(paths)):
		im1 = Image.open(paths[i])
		imgResize(im1,paths[i])
	
if __name__ == "__main__":
    main()
