# Copyright 2017 Christoph Baur <c.baur@tum.de>
# ==============================================================================


import sys
import os
import matplotlib.pyplot
import numpy as np
import tensorflow as tf
from MSSEG2008 import *


def conv_layer(input, filters, kernel_size=7):
    return tf.layers.conv2d(
            inputs = input,
            filters = filters,
            kernel_size = [kernel_size, kernel_size],
            padding = "same",
            activation = tf.nn.relu
            )


def deconv_layer(input,filters):
    return tf.layers.conv2d(
            inputs = input,
            filters = filters,
            kernel_size = [7,7],
            padding = "same",
            use_bias=False
            )


def maxpool_layer(input):
    return(tf.nn.max_pool_with_argmax(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'))


def unpool_layer(input, index, ksize=[1, 2, 2, 1]):
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


def model(input, scope="SegNet", reuse=True):
    #n_labels = 2
    with tf.name_scope(scope):
        # first layer
        conv_1 = conv_layer(input, 64)
        conv_2 = conv_layer(conv_1, 64)
        pool_1, indicies_1 = maxpool_layer(conv_2)

        # second layer
        conv_3 = conv_layer(pool_1, 128)
        conv_4 = conv_layer(conv_3, 128)
        pool_2, indicies_2 = maxpool_layer(conv_4)

        # third layer
        conv_5 = conv_layer(pool_2, 256)
        conv_6 = conv_layer(conv_5, 256)
        conv_7 = conv_layer(conv_6, 256)
        pool_3, indicies_3 = maxpool_layer(conv_7)

        # fourth layer
        conv_8 = conv_layer(pool_3, 512)
        conv_9 = conv_layer(conv_8, 512)
        conv_10 = conv_layer(conv_9, 512)
        pool_4, indicies_4 = maxpool_layer(conv_10)

        # fifth layer
        conv_11 = conv_layer(pool_4, 512)
        conv_12 = conv_layer(conv_11, 512)
        conv_13 = conv_layer(conv_12, 512)
        pool_5, indicies_5 = maxpool_layer(conv_13)

        # the decoding layers
        # fifth layer
        unpool_5 = unpool_layer(pool_5, indicies_5)
        deconv_13 = deconv_layer(unpool_5, 512)
        deconv_12 = deconv_layer(deconv_13, 512)
        deconv_11 = deconv_layer(deconv_12, 512)

        # forth layer
        unpool_4 = unpool_layer(deconv_11, indicies_4)
        deconv_10 = deconv_layer(unpool_4, 512)
        deconv_9 = deconv_layer(deconv_10, 512)
        deconv_8 = deconv_layer(deconv_9, 256)

        # third layer
        unpool_3 = unpool_layer(deconv_8, indicies_3)
        deconv_7 = deconv_layer(unpool_3, 256)
        deconv_6 = deconv_layer(deconv_7, 256)
        deconv_5 = deconv_layer(deconv_6, 128)

        # second layer
        unpool_2 = unpool_layer(deconv_5, indicies_2)
        deconv_4 = deconv_layer(unpool_2, 128)
        deconv_3 = deconv_layer(deconv_4, 64)

        # first layer
        unpool_1 = unpool_layer(deconv_3, indicies_1)
        deconv_2 = deconv_layer(unpool_1, 64)
        #deconv_1 = deconv_layer(deconv_2, 64)

        # Classification
        logits = deconv_layer(deconv_2, 1)
        softmax = tf.nn.softmax(logits)
        return logits, softmax


def printNumberOfTrainableParams():
    total_parameters = 0
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for variable in variables:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parametes = 1
        for dim in shape:
            #    print(dim)
            variable_parametes *= dim.value
        # print(variable_parametes)
        total_parameters += variable_parametes
    print(total_parameters)


def visualizeCurves(curves, handle=None):
    if not handle:
        handle = plt.figure()

    fig = plt.figure(handle.number)
    fig.clear()
    ax = plt.axes()
    plt.cla()

    counter = len(curves[list(curves.keys())[0]])
    x = np.linspace(0, counter, num=counter)
    for key, value in curves.items():
        value_ = np.array(value).astype(np.double)
        mask = np.isfinite(value_)
        ax.plot(x[mask], value_[mask], label=key)
    plt.legend(loc='upper right')
    plt.title("Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    #display.clear_output(wait=True)
    plt.savefig("accuracy.png")


dataset_options = MSSEG2008.Options()
dataset_options.debug = True
dataset_options.dir = '/media/data/MSSEG2008/'
dataset_options.useCrops = True
dataset_options.cropType = 'random'
dataset_options.cropWidth = 128
dataset_options.cropHeight = 128
dataset_options.numRandomCropsPerSlice = 10  # Not needed when doing center crops
dataset_options.onlyPatchesWithLesions = True
dataset_options.rotations = range(-5,5)
dataset_options.partition = {'TRAIN': 0.7, 'VAL': 0.3}
dataset_options.sliceResolution = None
dataset_options.cache = True
dataset_options.numSamples = -1
dataset_options.addInstanceNoise = False
dataset_options.axis = 'axial'
dataset_options.filterProtocol = ['FLAIR']
dataset_options.normalizationMethod = 'standardization'

# Center Crops of healthy control images: training, validation and testing patients
dataset = MSSEG2008(dataset_options)
config = {}
config['batchsize'] = 4
config['learningrate'] = 0.01
config['numEpochs'] = 10000

tf.reset_default_graph()

# Define placeholders
inputs = {}
inputs['data'] = tf.placeholder(tf.float32, [config['batchsize'], 128, 128, 1])
inputs['labels'] = tf.placeholder(tf.float32, [config['batchsize'], 128, 128, 1])
inputs['phase'] = tf.placeholder(tf.bool)

# Define a dictionary for storing curves
curves = {}
curves['training'] = []
curves['validation'] = []

logits, probabilities = model(inputs['data'])
printNumberOfTrainableParams()
onehot_labels = tf.one_hot(inputs['labels'], 2)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=onehot_labels))

correct_prediction = tf.equal(logits, inputs['labels'])
accuracy_operation = tf.cast(correct_prediction, tf.float32)

train_step = tf.train.GradientDescentOptimizer(config['learningrate']).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

numTrainSamples = dataset.numBatches(config['batchsize'], set='TRAIN')
numValSamples = dataset.numBatches(config['batchsize'], set='VAL')
numTestSamples = dataset.numBatches(config['batchsize'], set='TEST')
print(numTrainSamples)
for e in range(config['numEpochs']):
    avg_loss_in_current_epoch = 0
    for i in range(0, numTrainSamples):
        batch_data, batch_labels, _ = dataset.next_batch(config['batchsize'])
        print(batch_data.shape)
        batch_data = batch_data.reshape((batch_data.shape[0], 128, 128, 1))
        fetches = {
            'optimizer': train_step,
            'loss': cross_entropy
        }
        results = sess.run(fetches, feed_dict={inputs['data']: batch_data, inputs['labels']: batch_labels})
        avg_loss_in_current_epoch += results['loss']
    avg_loss_in_current_epoch = avg_loss_in_current_epoch / numTrainSamples
    print(avg_loss_in_current_epoch)
    curves['training'] += [avg_loss_in_current_epoch]

    for i in range(0, numValSamples):
        # Use Matplotlib to visualize the loss on the training and validation set
        batch_data, batch_labels, _ = dataset.next_batch(config['batchsize'], set='VAL')
        batch_data = batch_data.reshape((batch_data.shape[0], 128, 128, 1))
        fetches = {
            'loss': cross_entropy
        }
        results = sess.run(fetches, feed_dict={inputs['data']: batch_data, inputs['labels']: batch_labels})
        avg_loss_in_current_epoch += results['loss']
    avg_loss_in_current_epoch = avg_loss_in_current_epoch / numValSamples
    curves['validation'] += [avg_loss_in_current_epoch]

    print('Done with epoch %d' % (e))
    visualizeCurves(curves)
"""
accumulated_predictions = np.array([])
for i in range(0, numValSamples):
    # Use Matplotlib to visualize the loss on the training and validation set
    batch_data, batch_labels, _ = dataset.next_batch(config['batchsize'], set='TEST')
    batch_data = batch_data.reshape((batch_data.shape[0], 128, 128, 1))
    fetches = {
        'accuracy': accuracy_operation
    }
    results = sess.run(fetches, feed_dict={inputs['data']: batch_data, inputs['labels']: batch_labels})

    if i == 0:
        accumulated_predictions = results['accuracy']
    else:
        accumulated_predictions = np.append(accumulated_predictions, results['accuracy'])
accuracy = np.mean(accumulated_predictions)
print(accuracy)
"""
