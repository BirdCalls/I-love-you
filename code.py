import tensorflow as tf
import numpy as np
import math
# include any other imports that you want

'''
This file contains a class for you to implement your neural net.
Basic function skeleton is given, and some comments to guide you are also there.

You will find it convenient to look at the tensorflow API to understand what functions to use.
'''

""" useful things 

		tf.to_float(y)	
		loss  = tf.Print(loss, [loss], message="This is loss: ") -> when you run sess on loss, values get printed

"""

'''
Implement the respective functions in this class
You might also make separate classes for separate tasks , or for separate kinds of networks (normal feed-forward / CNNs)
'''
class myNeuralNet:
	# you can add/modify arguments of *ALL* functions
	# you might also add new functions, but *NOT* remove these ones
	def __init__(self, dim_input_data, dim_output_data): # you can add/modify arguments of this function 
		# Using such 'self-ization', you can access these members in later functions of the class
		# You can do such 'self-ization' on tensors also, there is no change
		self.dim_input_data = dim_input_data
		self.dim_output_data = dim_output_data
		self.x = tf.placeholder(tf.float32, shape=[None, dim_input_data])
		self.y = tf.placeholder(tf.float32, shape=[None, dim_output_data])

		# Create placeholders for input : data as well as labels
		# You might want to initialising some container to store all the layers of the network

		# own code
		tf.container('network')		# initialize container
		g = tf.container('network') # set g as the default graph

	def addHiddenLayer(self, input_layer, layer_dim, activation_fn=None, bias_regularizer_fn=None, kernel_regularizer_fn=None):

		output_layer = tf.layers.dense(inputs=input_layer, units=layer_dim, activation=activation_fn, bias_regularizer=bias_regularizer_fn,kernel_regularizer=kernel_regularizer_fn)

		return output_layer


	def addFinalLayer(self, input_layer, activation_fn=None, regularizer_fn=None,bias_regularizer_fn=None, kernel_regularizer_fn=None):
		# Create the output of the final layer as logits
		# You might also like to apply the final activation function (softmax / sigmoid) to get the predicted labels

		outer_layer = tf.layers.dense(inputs=input_layer, units=self.dim_output_data, activation=activation_fn, bias_regularizer=bias_regularizer_fn,kernel_regularizer=kernel_regularizer_fn)
											
		return outer_layer
	
	def setup_training(self, learn_rate, outer_layer):

		squared_delta = tf.square(outer_layer - tf.to_float(self.y)) # might not need to convert y into float
		self.loss = tf.reduce_sum(squared_delta)
		self.loss  = tf.Print(self.loss, [self.loss], message="This is loss: ") # this is so that it prints loss
		optimizer = tf.train.GradientDescentOptimizer(learn_rate)
		self.train_step = optimizer.minimize(self.loss)


	def setup_metrics(self):
		# Use the predicted labels and compare them with the input labels(placeholder defined in __init__)
		# to calculate accuracy, and store it as self.accuracy
		pass
	
	# you will need to add other arguments to this function as given below
	def train(self, sess, xtrain, ytrain, max_epochs, batch_size, train_size, print_step = 100): # valid_size, test_size, etc
		# Write your training part here
		# sess is a tensorflow session, used to run the computation graph
		# note that all the functions uptil now were just constructing the computation graph
		
		# one 'epoch' represents that the network has seen the entire dataset once - it is just standard terminology
		steps_per_epoch = int(train_size/batch_size)
		max_steps = max_epochs * steps_per_epoch
		for step in range(max_steps):
			# read a batch of data from the training data
			# now run the train_step, self.loss on this batch of training data. something like :
			_, train_loss = sess.run([self.train_step, self.loss], feed_dict={'''here, feed in your placeholders with the data you read in the previous to previous comment'''})
			if (step % print_step) == 0:
				# read the validation dataset and report loss, accuracy on it by running
				val_acc, val_loss = sess.run([self.accuracy, self.loss], feed_dict={'''here, feed in your placeholders with the data you read in the comment above'''})
				# remember that the above will give you val_acc, val_loss as numpy values and not tensors
				pass
			# store these train_loss and validation_loss in lists/arrays, write code to plot them vs steps
			# Above curves are *REALLY* important, they give deep insights on what's going on
		# -- for loop ends --
		# Now once training is done, run predictions on the test set
		test_predictions = sess.run('''here, put something like self.predictions that you would have made somewhere''', feed_dict={'''here, feed in test dataset'''})
		return test_predictions
		# This is because we will ask you to submit test_predictions, and some marks will be based on how your net performs on these unseen instances (test set)
		'''
		We have done everything in train(), but
		you might want to create another function named eval(),
		which calculates the predictions on test instances ...
		'''

	'''
	NOTE:
	you might find it convenient to make 3 different train functions corresponding to the three different tasks,
	and call the relevant one from each train_*.py
	The reason for this is that the arguments to the train() are different across the tasks
	'''
	'''
	Example, for the speech part, the train() would look something like :
	(NOTE: this is only a rough structure, we don't claim that this is exactly what you have to do.)
	
	train(self, sess, batch_size, train_size, max_epochs, train_signal, train_lbls, valid_signal, valid_lbls, test_signal):
		steps_per_epoch = math.ceil(train_size/batch_size)
		max_steps = max_epochs*steps_per_epoch
		print(max_steps)
		for step in range(max_steps):
			# select batch_size elements randomly from training data
			sampled_indices = random.sample(range(train_size), batch_size)
			trn_signal = train_signal[sampled_indices]
			trn_labels = train_lbls[sampled_indices]
			if (step % steps_per_epoch) == 0:
				val_loss, val_acc = sess.run([self.loss, self.accuracy], feed_dict={input_data: valid_signal, input_labels: valid_lbls})
				print(step, val_acc, val_loss)
			sess.run(self.train_step, feed_dict={input_data: trn_signal, input_labels: trn_labels})
		test_prediction = sess.run([self.predictions], feed_dict={input_data: test_signal})
		return test_prediction
	'''
