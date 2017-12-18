import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import misc
import pylab

def conv_layer(input, filters, kernel_size, strides = 1, padding = "same"):
    return tf.layers.conv2d(
            inputs = input,
            filters = filters,
            kernel_size = [kernel_size, kernel_size],
            padding = "same",
            activation = tf.nn.relu,
            strides = (strides, strides)
            )

def norm(l_input, lsize=4):
    return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75)


"""
def AlexNet(tensor):
    conv_1 = conv_layer(tensor, 96, 11, padding = "valid")
    pool_1 = tf.layers.max_pooling2d(inputs=conv_1, pool_size = [2,2], strides = 2, padding="valid")
    conv_2 = conv_layer(pool_1, 256, 5)
    pool_2 = tf.layers.max_pooling2d(inputs=conv_2, pool_size = [2,2], strides = 2, padding="valid")
    conv_3 = conv_layer(pool_2, 384, 3)
    conv_4 = conv_layer(conv_3, 384, 3)
    conv_5 = conv_layer(conv_4, 256, 3)
    pool_5 = tf.layers.max_pooling2d(inputs=conv_5, pool_size = [2,2], strides = 2, padding="valid")
    pool_5_flat = tf.reshape(pool_5, [-1, 4 * 4 * 256])
    dense_1 = tf.layers.dense(inputs=pool_5_flat, units = 120, activation=tf.nn.relu)
    dense_2 = tf.layers.dense(inputs=dense_1, units = 84, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense_2, units=10)
    return logits, tf.nn.softmax(logits)
"""


def AlexNet(tensor):
    conv_1 = conv_layer(tensor, 32, 11, padding="valid")
    pool_1 = tf.layers.max_pooling2d(inputs=conv_1, pool_size=[2, 2], strides=2, padding="valid")
    norm_1 = norm(pool_1, lsize=4)

    conv_2 = conv_layer(norm_1, 64, 5)
    pool_2 = tf.layers.max_pooling2d(inputs=conv_2, pool_size=[2, 2], strides=2, padding="valid")
    norm_2 = norm(pool_2, lsize=4)
    """
    conv_3 = conv_layer(norm_2, 256, 3)
    #conv_4 = conv_layer(conv_3, 384, 3)
    #conv_5 = conv_layer(conv_4, 256, 3)
    pool_5 = tf.layers.max_pooling2d(inputs=conv_3, pool_size = [2,2], strides = 2, padding="valid")
    norm_5 = norm(pool_5, lsize=4)
    """
    pool_5_flat = tf.reshape(norm_2, [-1, 8 * 8 * 64])
    dense_1 = tf.layers.dense(inputs=pool_5_flat, units=120, activation=tf.nn.relu)
    # dense_2 = tf.layers.dense(inputs=dense_1, units = 84, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense_1, units=10)
    return logits, tf.nn.softmax(logits)

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
    #plt.show()
    plt.savefig('foo.png')

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

config = {}
config['batchsize'] = 128
config['learningrate'] = 0.01
config['numEpochs'] = 10
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print("Image Shape: {}".format(mnist.train.images[0].shape))
print()
print("Training Set:   {} samples".format(len(mnist.train.images)))
print("Validation Set: {} samples".format(len(mnist.validation.images)))
print("Test Set:       {} samples".format(len(mnist.test.images)))

# Define placeholders
tf.reset_default_graph()
inputs = {}
inputs['data'] = tf.placeholder(tf.float32, [None, 32, 32, 1])
inputs['labels'] = tf.placeholder(tf.float32, [None, 10])
inputs['phase'] = tf.placeholder(tf.bool)

# Define a dictionary for storing curves
curves = {}
curves['training'] = []
curves['validation'] = []

# Instantiate the model operations
logits, probabilities = AlexNet(inputs['data']) # Or VGGNet or ResNet or AlexNet
printNumberOfTrainableParams()

# Define loss function in a numerically stable way
# DONT: cross_entropy = tf.reduce_mean(-tf.reduce_sum( * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = inputs['labels']))

# Operations for assessing the accuracy of the classifier
correct_prediction = tf.equal(tf.argmax(probabilities,1), tf.argmax(inputs['labels'],1))
accuracy_operation = tf.cast(correct_prediction, tf.float32)

# Idea: Use different optimizers?
# SGD vs ADAM
train_step = tf.train.AdamOptimizer(config['learningrate']).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(config['learningrate']).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

numTrainSamples = len(mnist.train.images)
numValSamples = len(mnist.validation.images)
numTestSamples = len(mnist.test.images)
for e in range(config['numEpochs']):
    avg_loss_in_current_epoch = 0
    for i in range(0, numTrainSamples, config['batchsize']):
        batch_data, batch_labels = mnist.train.next_batch(config['batchsize'])
        batch_data = batch_data.reshape((batch_data.shape[0], 28, 28, 1))

        batch_data = np.pad(batch_data, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
        fetches = {
            'optimizer': train_step,
            'loss': cross_entropy
        }
        results = sess.run(fetches, feed_dict={inputs['data']: batch_data, inputs['labels']: batch_labels})
        avg_loss_in_current_epoch += results['loss']
    avg_loss_in_current_epoch = avg_loss_in_current_epoch / i
    print("Training", avg_loss_in_current_epoch)
    curves['training'] += [avg_loss_in_current_epoch]

    for i in range(0, numValSamples, config['batchsize']):
        # Use Matplotlib to visualize the loss on the training and validation set
        batch_data, batch_labels = mnist.validation.next_batch(config['batchsize'])
        batch_data = batch_data.reshape((batch_data.shape[0], 28, 28, 1))

        # TODO: Preprocess the images in the batch
        batch_data = np.pad(batch_data, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

        fetches = {
            'loss': cross_entropy
        }
        results = sess.run(fetches, feed_dict={inputs['data']: batch_data, inputs['labels']: batch_labels})
        avg_loss_in_current_epoch += results['loss']
    avg_loss_in_current_epoch = avg_loss_in_current_epoch / i
    print("Validation", avg_loss_in_current_epoch)
    curves['validation'] += [avg_loss_in_current_epoch]

    print('Done with epoch %d' % (e))
    visualizeCurves(curves)

accumulated_predictions = np.array([])
for i in range(0, numValSamples, config['batchsize']):
    # Use Matplotlib to visualize the loss on the training and validation set
    batch_data, batch_labels = mnist.test.next_batch(config['batchsize'])
    batch_data = batch_data.reshape((batch_data.shape[0], 28, 28, 1))

    # TODO: Preprocess the images in the batch
    batch_data = np.pad(batch_data, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
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
