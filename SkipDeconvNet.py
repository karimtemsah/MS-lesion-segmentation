# Copyright 2017 Christoph Baur <c.baur@tum.de>
# ==============================================================================


# Basic imports
import sys
import os
import matplotlib.pyplot
import numpy as np
import tensorflow as tf

def create_weights(filter_size, in_channels, out_channels):
	w = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], stddev=0.05)
	return tf.Variable(w)

def create_bias(num_filters):
	b = tf.constant(0.05, shape=[num_filters])
	return tf.Variable(b)

def create_conv_layer(input, W, b, strides=[1, 1, 1, 1], padding='SAME'):
	conv = tf.nn.conv2d(input, W, strides, padding)
	return tf.nn.relu(conv + b)

def create_maxpool_layer(input):
	return(tf.nn.max_pool_with_argmax(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'))

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

def model(input, scope="SkipDeconvNet", reuse=True):

	with tf.name_scope(scope):

	    conv_1 = create_conv_layer(input, create_weights(7,1,64), create_bias(64)) 
	    maxpool_1, max_pool_1_index = create_maxpool_layer(conv_1) 

	    conv_2 = create_conv_layer(maxpool_1, create_weights(7,64,64), create_bias(64))
	    maxpool_2, max_pool_2_index = create_maxpool_layer(conv_2)

	    conv_3 = create_conv_layer(maxpool_2, create_weights(7,64,64), create_bias(64))
	    maxpool_3, max_pool_3_index = create_maxpool_layer(conv_3)

	    conv_4 = create_conv_layer(maxpool_3, create_weights(7,64,64), create_bias(64))

	    unpool_1 = create_unpool_layer(conv_4, max_pool_3_index)
	    unpool_1 = tf.concat([unpool_1, conv_3], axis=3)

	    deconv = create_conv_layer(unpool_1, create_weights(7,128,64), create_bias(64))
	    print(deconv.shape)

x = tf.placeholder(tf.float32, shape=(1, 256, 256, 1))

model(x)

