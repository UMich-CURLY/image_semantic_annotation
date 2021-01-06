import numpy as np
import glob
import cv2
import os

path_to_imgs = '/home/luoxin/Documents/cheetah/dataset/2012-03-17_lb3/labels_rgb_copy/'
paths = glob.glob(os.path.join(path_to_imgs, '*.png'))
paths.sort()

label_list = {tuple([ 33,  28,  29]):'0', 
              tuple([160, 235, 208]):'1',
              tuple([ 21, 237,  43]):'2',
              tuple([ 17, 240, 217]):'3', 
              tuple([ 65,  24, 186]):'4',
              tuple([ 28,   9, 237]):'5',
              tuple([ 98,  45, 235]):'6', 
              tuple([143,  99,  20]):'7',
              tuple([194, 199, 157]):'8',
              tuple([ 55,  61, 237]):'9', 
              tuple([232,  39,  32]):'10',
              tuple([245, 193,  37]):'11',
              tuple([127, 143, 132]):'12', 
              tuple([209, 151,  25]):'13',
              tuple([169,  90,  83]):'14',
              tuple([ 62, 163, 158]):'15', 
              tuple([127,  55, 182]):'16',
              tuple([173,  28, 101]):'17',
              tuple([104, 168, 162]):'18', 
              tuple([176, 135, 162]):'19',
              tuple([238, 149,  45]):'20'
              }

for path in paths:
	print(path)
	img = cv2.imread(path)
	height = img.shape[0]
	width = img.shape[1]
	channel = img.shape[2]
	for h in range(height):
		for w in range(width):
			ID = int(label_list[tuple(img[h][w])])
			for c in range(channel):
				img[h][w][c] = ID
	cv2.imwrite(path, img)