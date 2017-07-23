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
#########################################################################################
"""
//TRAINING HOG DESCRIPTOR :
//Extract the downloaded data, into the folder - 'new_train_data' in current folder

							< OR >

//If the data is not of logos as downloaded, make the data folders classwise as needed 
//First run the other code - 'prepare_data.py'; for this, create the data as needed 
  in the folder - 'raw_train_data', in current folder
//The main 'raw_data_folder', must contain sub-folders of classes (ex: BRANDS),
then each of these folders must contain sub-folders of labels (ex: MODELS),
and then the relevant data must be in the respective folders
//The data for training will be saved in the 'new_train_data'

//The default paths can be changed in the beginning of each code
//But, this is not recommended

//Run the train.py code as follows:
1) For training data :
	run 'train.py train'
	
	When prompted,
	For Logistic Classification: enter 'LOGIST'
	For Linear SVM MulitClass classification : enter 'SVM'

2) For classifying based on folder names in the 'new_train_data' :
	run 'train.py classify'

	When prompted,
	For Logistic Classification: enter 'LOGIST'
	For Linear SVM MulitClass classification : enter 'SVM'

	The relevant mode must be selected based on the mode used in training data

########################################################################################
The code works as follows:
1) The read_train_data_paths() function reads the data from the default data path
2) Cache is created only once, so that hog features are not created each time, code is run
3) Then the training starts..
4) The weights are saved in the folder 'hog_saved_weights', once the training is complete.
5) For classification, weights are retrieved from this folder
6) 'logo.jpg' file is used for testing the saved the weights. Make sure this file
is present in the current folder.

Change the training steps below, if required.

"""
###########################################################################
#num_classes will be length of class_list
num_classes = 0
training_steps = 1500
###########################################################################

import os, sys
import matplotlib.pyplot as plt
import matplotlib.image as iread
import tensorflow as tf
from PIL import Image
import numpy as np
from random import shuffle
from random import randint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cwd = os.getcwd()

if cwd == os.path.dirname(os.path.abspath(__file__)):
	import hog
else:
	folder = os.path.dirname(os.path.realpath(__file__))
	from HOG import hog

image_path = os.path.join(cwd,'logo.jpg')
train_path = os.path.join(cwd,'new_train_data')
save_path = os.path.join(cwd,'hog_saved_weights')
hog_file_path = os.path.join(cwd,'hog_files')
class_list = []
train_list = []
hog_list = []
total_data = 0
batch_size = 100
class_data_count = []

##########################################################################################
#reads train data images and makes lists of paths
def read_train_data_paths(num_classes,total_data):
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(hog_file_path):
    	os.makedirs(hog_file_path)
 	
    class_list.extend(os.listdir(train_path))
    num_classes += len(class_list)

    for folder,val in enumerate(class_list):
        class_path = os.path.join(train_path,val)
        hog_path = os.path.join(hog_file_path,val)

        if not os.path.exists(hog_path):
        	os.makedirs(hog_path)

        image_list = os.listdir(class_path)
        class_data_count.append(len(image_list))

        for i in image_list:
            img_path = os.path.join(class_path,i)
            train_list.append([img_path,folder])

            #makes paths for cache
            i = i.replace('.jpg','.txt')
            i = i.replace('.JPG','.txt')
            i = i.replace('.jpeg','.txt')
            i = i.replace('.JPEG','.txt')

            i = os.path.join(hog_path,i)
            hog_list.append([i,folder])

    total_data += len(hog_list)

    return num_classes,total_data

#creates cache in the form of .txt files
def create_cache():
	for index,image in enumerate(train_list):
		if not os.path.exists(hog_list[index][0]):
			#the following function is imported from hog file
			hog.create_hog_file(image[0],hog_list[index][0])
		else:
			print('Found cache... '+hog_list[index][0])

#Creates the variables for weights and biases
def create_variables(num_classes):
	W = tf.Variable(tf.truncated_normal([288, num_classes]),name='weights')
	b = tf.Variable(tf.truncated_normal([1, num_classes]), name='biases')

	return W,b

#creates labels; uses hog descriptors 
def create_labels(count, hog_list, total_data, batch_size):
	
	#labels are one-hot vectors. But 0 is replaced with -1
	point = count
	path = hog_list[count][0]
	lab = hog_list[count][1]
	y = np.zeros([1,num_classes])
	y[0][lab] = 1

	x = hog.read_hog_file(path)
	x = np.expand_dims(x,axis=0)
	
	count += 1
	extra = np.min([batch_size,total_data-point])

	while count<point+extra and count<total_data:
		path = hog_list[count][0]
		lab = hog_list[count][1]

		y_new = np.zeros([1,num_classes])
		y_new[0][lab] = 1
		y = np.concatenate((y,y_new), axis=0)

		x_new = hog.read_hog_file(path)
		x_new = np.expand_dims(x_new,axis=0)
		x = np.concatenate((x,x_new), axis=0)

		count+=1

	return x,y

#evaluates accuracy
def evaluate_accuracy(final,labels):
	prediction = tf.argmax(final,axis=1)
	ground_truth = tf.argmax(labels,axis=1)

	evaluate = tf.equal(prediction,ground_truth)
	accuracy = tf.reduce_mean(tf.cast(evaluate,dtype=tf.float32), axis=0)

	return accuracy*100

#Creates a model for SOFTMAX 
def model(W,b,num_classes):
	x = tf.placeholder(tf.float32,[None, 288])
	y = tf.placeholder(tf.float32,[None, num_classes])

	logits = tf.add(tf.matmul(x,W),b)
	prediction = tf.nn.softmax(logits)

	loss = tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=logits)
	loss = tf.reduce_mean(loss)

	optimizer = tf.train.AdamOptimizer()
	train_step = optimizer.minimize(loss)
	accuracy = evaluate_accuracy(prediction,y)

	return train_step,accuracy,x,y
 
#training in SOFTMAX Logistic mode
def train_values():
	W,b = create_variables(num_classes)
	train_step, accuracy,x,y = model(W,b,num_classes)
	
	print('\n--------------------------------------------------------------------')
	print('ONE v/s ALL training - SOFTMAX LOGISTIC MULTICLASSIFIER')
	print('--------------------------------------------------------------------')

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for epoch in range(training_steps):
			print('\nTraining step : '+str(epoch+1)+' .....................')	
			count = 0
			while count<total_data:
				X,Y = create_labels(count, hog_list, total_data, batch_size)
				
				_,accu = sess.run([train_step,accuracy],feed_dict={x:X,y:Y})
				print('Batch training Accuracy : '+str(accu)+' ...')

				extra = np.min([batch_size,total_data-count])
				count += extra

		#saving weights
		write_ckpt(W,sess,'weights','LOGIST')
		write_ckpt(b,sess,'biases','LOGIST')

		weight = sess.run(W)
		bias = sess.run(b)

	#Here, test data is randomly selected from the main data set
	k = int(0.1*(len(hog_list)))
	test = generate_random_test(k)
	X,Y = create_labels(0, test, k, k)
	_,pred = classify(X, weight.astype(dtype=np.float32), bias.astype(dtype=np.float32))
	accu = evaluate_accuracy(pred, Y)

	#Accuracy for test
	with tf.Session() as sess:
		print('\nTest Accuracy : '+str(sess.run(accu))+' % ....')

	return weight,bias

#Classifying using Logistic function
def classify(X,W,b):
	batch = X.shape[0]
	X = tf.convert_to_tensor(X,dtype=tf.float32)
	logits = tf.add(tf.matmul(X,W),b)
	y = tf.nn.softmax(logits)
	#score is the maximum probability obtained by the classifier
	score = tf.reduce_max(y, axis=1)
	
	with tf.Session() as sess:
		num = sess.run(tf.argmax(y,axis=1))
		score = sess.run(score)
	
	#creating label for calculating accuracy	
	prediction = np.zeros([batch,num_classes])
	
	for i in range(batch):
		prediction[i][num[i]] = 1

	return score,prediction

#Saves weights to file
def write_ckpt(tensor, sess, name, mode):
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	#saves weights in the respective mode folder
	mode_path = os.path.join(save_path,mode)
	if not os.path.exists(mode_path):
		os.makedirs(mode_path)

	folder_path = os.path.join(mode_path,name)
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)

	#saves as a .ckpt file
	saver = tf.train.Saver({name:tensor})
	filename = name+'.ckpt'
	path = os.path.join(folder_path,filename)
	tensor_path = saver.save(sess, path)

	print("\nHog tensor saved at %s", tensor_path)

#reads .ckpt file and restores variables
#Variables must be created before calling this
def read_ckpt(ckpt_path,name,tensor,sess):
	saver = tf.train.Saver({name:tensor})
	saver.restore(sess, ckpt_path)

#Creating SVM labels
#key for SVM is taken -1 here
def create_svm_labels(count, hog_list, total_data, batch_size, class_num, key):
	point = count
	path = hog_list[count][0]
	lab = hog_list[count][1]

	y = np.array([[key]])
	if lab==class_num:
		y[0][0] = 1

	x = hog.read_hog_file(path)
	x = np.expand_dims(x,axis=0)
	
	count += 1
	extra = np.min([batch_size,total_data-point])

	while count<point+extra and count<total_data:
		path = hog_list[count][0]
		lab = hog_list[count][1]
		
		y_new = np.array([[key]])
		if lab==class_num:
			y_new[0][0] = 1

		y = np.concatenate((y,y_new), axis=0)
		
		x_new = hog.read_hog_file(path)
		x_new = np.expand_dims(x_new,axis=0)
		x = np.concatenate((x,x_new), axis=0)

		count+=1

	return x,y

#Creates Linear SVM Model
def Linear_SVM_model(W,b):
	#W must be of shape [288,1]
	x = tf.placeholder(tf.float32,[None, 288])
	y = tf.placeholder(tf.float32,[None, 1])

	# Regularisation constant
	C = 1

	# Model is as follows:
	# hyperplane : hplane = W*x + b
	# cost = (1/n)*sum( max( 0, (1-y*hplane) ) ) + C*||W||^2
	h_plane = tf.add(tf.matmul(x,W),b)
	h_plane = 1.-tf.multiply(y,h_plane)
	cost = tf.maximum(0.,h_plane)
	cost = tf.reduce_mean(cost,axis=0)
	cost += C*tf.reduce_sum(tf.square(W), axis=1)

	optimizer = tf.train.AdamOptimizer()
	train_step = optimizer.minimize(cost)

	return train_step,x,y

#Generates random test data from the main data list
#num is the number of data
def generate_random_test(num):
	test = []
	
	for i in range(num):
		s = randint(0,total_data)
		test.append(hog_list[s])

	return test

#Trains SVM model
#Training each class separately
#One vs All classification
def train_SVM():
	print('\n--------------------------------------------------------------------')
	print('ONE v/s ALL training - SVM MULTICLASSIFIER')
	print('--------------------------------------------------------------------')

	W_main = np.zeros([288,num_classes])
	b_main = np.zeros([1,num_classes])
	for i in range(num_classes):
		W = tf.Variable(tf.truncated_normal([288,1]))
		b = tf.Variable(tf.truncated_normal([1,1]))
		print('\nTraining SVM for Class '+str(i+1)+'/'+str(num_classes)+' : ' + class_list[i]+' .......................................\n')
		train_step,x,y = Linear_SVM_model(W,b)
		
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for epoch in range(training_steps):
				print('................ '+str(i+1)+'/'+str(num_classes)+' Training step : '+str(epoch+1)+' ................')
				count = 0
				while count<total_data:
					print('Image: '+str(count+1)+'/'+str(total_data)+' ...')
					X,Y = create_svm_labels(count, hog_list, total_data, batch_size, i, -1)
					sess.run(train_step,feed_dict={x:X,y:Y})
					
					extra = np.min([batch_size,total_data-count])
					count += extra
			
			#Weights for each class are added to the main matrix as columns
			W_main[:,i] = (sess.run(W))[:,0]
			b_main[:,i] = (sess.run(b))[:,0]

	#Generates Test data and tests the trained model
	k = int(0.1*(len(hog_list)))
	test = generate_random_test(k)
	X,Y = create_labels(0, test, k, k)
	_,_,pred = SVM_classify(X, W_main.astype(dtype=np.float32), b_main.astype(dtype=np.float32))
	accu = evaluate_accuracy(pred, Y)
	with tf.Session() as sess:
		print('\nTest Accuracy : '+str(sess.run(accu))+' % ....')

	#Creates weights and biases for saving
	W_final = tf.Variable(W_main.astype(dtype=np.float32),name='weights')
	b_final = tf.Variable(b_main.astype(dtype=np.float32),name='biases')
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		write_ckpt(W_final,sess,'weights','SVM')
		write_ckpt(b_final,sess,'biases','SVM')

	return W_main,b_main

#Classifier for SVM Model
def SVM_classify(X,W,b):
	batch = X.shape[0]
	X = tf.convert_to_tensor(X,dtype=tf.float32)
	h_plane = tf.add(tf.matmul(X,W),b)
	#score is the maximum positive distance from the hyperplane
	score = tf.reduce_max(h_plane, axis=1)

	with tf.Session() as sess:
		num = sess.run(tf.argmax(h_plane,axis=1))
		score = sess.run(score)
		plane = sess.run(h_plane)

	#Creating label vector for validating accuracy
	prediction = np.zeros([batch,num_classes])
	for i in range(batch):
		prediction[i][num[i]] = 1
	
	return score,plane, prediction

##################################################################################################

if __name__ == '__main__':

	if sys.argv[1]=='train':

		line = input('Enter the mode of training ( SVM / LOGIST ) : ?\n')

		#in case of logos, num_classes is the no. of Brands
		a,b = read_train_data_paths(num_classes,total_data)
		num_classes += a
		total_data += b

		#Saving cache in the form of txt file
		#Hog features are saved as cache
		create_cache()
		
		print('\nTotal '+str(total_data)+ ' images are found.....')
		print('\nTraining Steps :  '+str(training_steps)+'\n')

		#checking the mode input
		if line=='SVM':		
			weights,biases = train_SVM()

			W_tensor = tf.convert_to_tensor(weights, dtype=tf.float32)
			b_tensor = tf.convert_to_tensor(biases, dtype=tf.float32)

			#hog module is called here again
			X = hog.hog_from_path(image_path)
			_,_,prediction = SVM_classify(X,W_tensor,b_tensor)

		elif line=='LOGIST':
			shuffle(hog_list)

			W_new,b_new = train_values()

			W_tensor = tf.convert_to_tensor(W_new, dtype=tf.float32)
			b_tensor = tf.convert_to_tensor(b_new, dtype=tf.float32)
			
			#hog module is called here again
			X = hog.hog_from_path(image_path)
			_,prediction = classify(X,W_tensor,b_tensor)
		
		print('\nThe logo belongs to : '+str(class_list[np.argmax(prediction)]))
		
	elif sys.argv[1]=='classify':
		line = input('Enter mode of training used ( SVM / LOGIST ) : ?\n')

		#in case of logos, num_classes is the no. of brands
		a,b = read_train_data_paths(num_classes,total_data)
		num_classes += a
		total_data += b

		W,b = create_variables(num_classes)
		mode_path = os.path.join(save_path, line)

		#reading weights from the saved checkpoints
		with tf.Session() as sess:
			read_ckpt(os.path.join(mode_path,'weights/weights.ckpt'),'weights',W,sess)
			read_ckpt(os.path.join(mode_path,'biases/biases.ckpt'),'biases',b,sess)
			
			W_array = sess.run(W)
			b_array = sess.run(b)

		W_array = tf.convert_to_tensor(W_array, dtype=tf.float32)
		b_array = tf.convert_to_tensor(b_array, dtype=tf.float32)

		#Extracting Hog features
		X = hog.hog_from_path(image_path)
		
		#Classifying using mode
		if line=='SVM':
			_,_,prediction = SVM_classify(X,W_array,b_array)
		elif line=='LOGIST':
			_,prediction = classify(X,W_array,b_array)

		print('\nThe logo belongs to : '+str(class_list[np.argmax(prediction)]))

	else:
		print('ERROR :: Enter either train or classify after file name......!!')

