from __future__ import print_function

import tensorflow as tf
import math
import numpy as np

from model import myNeuralNet

# x denotes features, y denotes labels
xtrain = np.load('data/mnist/xtrain.npy')
ytrain = np.load('data/mnist/ytrain.npy') #ytrain is a lable like [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]

xval = np.load('data/mnist/xval.npy') # val stands for validation 
yval = np.load('data/mnist/yval.npy') #yval is a lable like [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]

xtest = np.load('data/mnist/xtest.npy')

dim_input = 784 # number of input numbers
dim_output = 10
layer_dim1 =1000;
layer_dim2 =1000;

max_epochs = 10
learn_rate = 0.01
batch_size = 50 # After 50 batches, variables(w,b) get's updated. it's too big to update values after all 55300 cases. 

train_size = len(xtrain) # there are 55300 cases 
valid_size = len(xval)
test_size = len(xtest)

total_images = []
total_labels = []


# Create Computation Graph
nn_instance = myNeuralNet(dim_input, dim_output) 
layer_1 = nn_instance.addHiddenLayer(nn_instance.x,layer_dim1,activation_fn=tf.nn.sigmoid)
layer_2 = nn_instance.addHiddenLayer(layer_1,layer_dim2,activation_fn=tf.nn.sigmoid)

outer_layer = nn_instance.addFinalLayer(layer_2,activation_fn=tf.nn.softmax) # should use softmax? 
nn_instance.setup_training(learn_rate, outer_layer)
nn_instance.setup_metrics()  # what is this doing? 
# Training steps

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	test_pred = nn_instance.train(sess, xtrain, ytrain, max_epochs,batch_size,train_size) # fill in other arguments as you modify the train(self, sess, ...) in model.py
	# you will have to pass xtrain, ytrain, etc ... also as arguments so that you can sample batches in train() of model.py

# write code here to store test_pred in relevant file
	
