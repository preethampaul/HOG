# Copyright 2017 Sunkari Preetham Paul. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################################
"""
//HOG FEATURES EXTRACTION :
Hog descriptor uses edge detection by gradient calculation and histograms of gradients,
with magnitudes as weights

The code uses [-1 0 -1] kernel for gradient magnitude and orientation calculation
Gradients are calculated in the range [0,180]
Histograms of 8 bins are calculated with magnitudes as weights 
Each image is checked if its of 32X32 size, else its resized
The code reads images in greyscale.

The images are normalised for gamma, and then, for normal contrast
Each 32X32 image pixel matrix, is organised into 8X8 cells and then, histograms
are calculated for each cell. Then, a 4X4 matrix with 8 bins in each cell is obtained

This matrix is organised as 2X2 blocks(with 50% overlap) and normalised, by dividing with the 
magnitude of histogram bins' vector.
A total of 9 blocks X 4 cells X 8 bins  = 288 features

"""
#################################################################################################

import os, sys
import matplotlib.pyplot as plt
import matplotlib.image as iread
import tensorflow as tf
from PIL import Image
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cwd = os.getcwd()

#for sliding window for calculating histogram
#stide = 50, incr stands for this
cell = [8, 8]
incr = [8,8]
bin_num = 8
im_size = [32,32]


#image path must be wrt current working directory
def create_array(image_path):
	
	image = Image.open(os.path.join(cwd,image_path)).convert('L')
	image_array = np.asarray(image,dtype=float)
	
	return image_array

#uses a [-1 0 1 kernel]
def create_grad_array(image_array):
	image_array = Image.fromarray(image_array)
	if not image_array.size == im_size:
		image_array = image_array.resize(im_size, resample=Image.BICUBIC)
	
	image_array = np.asarray(image_array,dtype=float)
	
	# gamma correction
	image_array = (image_array)**2.5

	# local contrast normalisation
	image_array = (image_array-np.mean(image_array))/np.std(image_array)
	max_h = 32
	max_w = 32

	grad = np.zeros([max_h, max_w])
	mag = np.zeros([max_h, max_w])
	for h,row in enumerate(image_array):
		for w, val in enumerate(row):
			if h-1>=0 and w-1>=0 and h+1<max_h and w+1<max_w:
				dy = image_array[h+1][w]-image_array[h-1][w]
				dx = row[w+1]-row[w-1]+0.0001
				grad[h][w] = np.arctan(dy/dx)*(180/np.pi)
				if grad[h][w]<0:
					grad[h][w] += 180
				mag[h][w] = np.sqrt(dy*dy+dx*dx)
	
	return grad,mag

def write_hog_file(filename, final_array):
	print('Saving '+filename+' ........\n')
	np.savetxt(filename,final_array)

def read_hog_file(filename):
	return np.loadtxt(filename)

def calculate_histogram(array,weights):
	bins_range = (0, 180)
	bins = bin_num
	hist,_ = np.histogram(array,bins=bins,range=bins_range,weights=weights)

	return hist

def create_hog_features(grad_array,mag_array):
	max_h = int(((grad_array.shape[0]-cell[0])/incr[0])+1)
	max_w = int(((grad_array.shape[1]-cell[1])/incr[1])+1)
	cell_array = []
	w = 0
	h = 0
	i = 0
	j = 0

	#Creating 8X8 cells
	while i<max_h:
		w = 0
		j = 0

		while j<max_w:
			for_hist = grad_array[h:h+cell[0],w:w+cell[1]]
			for_wght = mag_array[h:h+cell[0],w:w+cell[1]]
			
			val = calculate_histogram(for_hist,for_wght)
			cell_array.append(val)
			j += 1
			w += incr[1]

		i += 1
		h += incr[0]

	cell_array = np.reshape(cell_array,(max_h, max_w, bin_num))
	#normalising blocks of cells
	block = [2,2]
	#here increment is 1

	max_h = int((max_h-block[0])+1)
	max_w = int((max_w-block[1])+1)
	block_list = []
	w = 0
	h = 0
	i = 0
	j = 0

	while i<max_h:
		w = 0
		j = 0

		while j<max_w:
			for_norm = cell_array[h:h+block[0],w:w+block[1]]
			mag = np.linalg.norm(for_norm)
			arr_list = (for_norm/mag).flatten().tolist()
			block_list += arr_list
			j += 1
			w += 1

		i += 1
		h += 1

	#returns a vextor array list of 288 elements
	return block_list

#image_array must be an array
#returns a 288 features vector from image array
def apply_hog(image_array):
	gradient,magnitude = create_grad_array(image_array)
	hog_features = create_hog_features(gradient,magnitude)
	hog_features = np.asarray(hog_features,dtype=float)
	hog_features = np.expand_dims(hog_features,axis=0)

	return hog_features

#path must be image path
#returns final features array from image_path
def hog_from_path(image_path):
	image_array = create_array(image_path)
	final_array = apply_hog(image_array)
	
	return final_array

#Creates hog files
def create_hog_file(image_path,save_path):
	image_array = create_array(image_path)
	final_array = apply_hog(image_array)
	write_hog_file(save_path,final_array)

if __name__ == '__main__':
	create_hog_file('logo.jpg','logo.txt')
	mg = read_hog_file('logo.txt')
	print(mg)
	print(mg.shape)