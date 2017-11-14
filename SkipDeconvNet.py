# Copyright 2017 Christoph Baur <c.baur@tum.de>
# ==============================================================================


# Basic imports
import sys
import os
import matplotlib.pyplot
import numpy as np
import tensorflow as tf
import cv2

#Initialize weights for the network
#@param filter_size : Size of the convolutional filter
#@param in_channels : Number of channels of the input image
#@param out_channels : Number of channels of the output image
def create_weights(filter_size, in_channels, out_channels):
	w = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], stddev=0.05)
	return tf.Variable(w)

#Initialize biases for the network
#@param num_filters : Num of convolutional filters
def create_bias(num_filters):
	b = tf.constant(0.05, shape=[num_filters])
	return tf.Variable(b)

#Convolutional net, batch normalisation, and ReLU
#@param input : Input tensor to the CNN
#@param W : Weights of the convolutional layer
#@param b : Biases of the input net
#@param strides : The stride of the convolution in each axis
#@param padding : The type of padding implemented
def create_conv_layer(input, W, b, strides=[1, 1, 1, 1], padding='SAME'):
	conv = tf.nn.conv2d(input, W, strides, padding)
	batch_mean,batch_var = tf.nn.moments(conv, axes = [0])
	scale = tf.Variable(tf.ones(conv.shape))
	offset = tf.Variable(tf.zeros(conv.shape))
	bn_conv = tf.nn.batch_normalization(conv, batch_mean, batch_var, offset, scale, np.finfo(float).eps)
	return tf.nn.relu(conv + b)

#Implements max pooling
#@param input : Input tensor to the max pooling layer
def create_maxpool_layer(input):
	return(tf.nn.max_pool_with_argmax(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'))

#Implements unpooling
#@param input : Input tensor to the unpooling layer
#@param index : Indices of the max positions
#@param ksize : Size of unpooling kernel
def create_unpool_layer(input, index, ksize=[1, 2, 2, 1]):
	input_shape = input.get_shape().as_list()
	output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

	flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

	uppool = tf.reshape(input, [np.prod(input_shape)])
	batch_range = tf.reshape(tf.range(output_shape[0], dtype=index.dtype), shape=[input_shape[0], 1, 1, 1])
	b = tf.ones_like(index) * batch_range
	b = tf.reshape(b, [np.prod(input_shape), 1])
	ind = tf.reshape(index, [np.prod(input_shape), 1])
	ind = tf.concat([b, ind], 1)

	output = tf.scatter_nd(ind, uppool, shape=flat_output_shape)
	return tf.reshape(output, output_shape)

#Model which implements SkipDeconvNet
#@param input : Image to be put through the net
#@param scope : Scope of the model
#@param reuse : If the model can be reused
def model(input, scope="SkipDeconvNet", reuse=True):

	N = 5

	with tf.name_scope(scope):

		#Model architecture
	    conv_1 = create_conv_layer(input, create_weights(7,1,64), create_bias(64)) 
	    maxpool_1, max_pool_1_index = create_maxpool_layer(conv_1) 

	    conv_2 = create_conv_layer(maxpool_1, create_weights(7,64,64), create_bias(64))
	    maxpool_2, max_pool_2_index = create_maxpool_layer(conv_2)

	    conv_3 = create_conv_layer(maxpool_2, create_weights(7,64,64), create_bias(64))
	    maxpool_3, max_pool_3_index = create_maxpool_layer(conv_3)

	    conv_4 = create_conv_layer(maxpool_3, create_weights(7,64,64), create_bias(64))

	    unpool_1 = create_unpool_layer(conv_4, max_pool_3_index)
	    unpool_1 = tf.concat([unpool_1, conv_3], axis=3)

	    deconv_1 = create_conv_layer(unpool_1, create_weights(7,128,64), create_bias(64))

	    unpool_2 = create_unpool_layer(deconv_1, max_pool_2_index)
	    unpool_2 = tf.concat([unpool_2, conv_2], axis=3)

	    deconv_2 = create_conv_layer(unpool_2, create_weights(7, 128, 64), create_bias(64))

	    unpool_3 = create_unpool_layer(deconv_2, max_pool_1_index)
	    unpool_3 = tf.concat([unpool_3, conv_1], axis=3)

	    deconv_3 = create_conv_layer(unpool_3, create_weights(7, 128, 64), create_bias(64))

	    #classifier layer
	    classifier_layer = create_conv_layer(deconv_3, create_weights(1, 64, N), create_bias(N))
	    softmax = tf.nn.softmax(classifier_layer)
	    print(softmax.shape)

#Main function to load image
im = cv2.imread('liver.jpeg', cv2.IMREAD_GRAYSCALE)
im = cv2.resize(im,(256,256))
im = np.array(im, dtype=np.float32)
im = np.expand_dims(im, axis = 0)
im = np.expand_dims(im, axis = 3)
x = tf.placeholder(tf.float32, shape=(1, 256, 256, 1))

model(im)

