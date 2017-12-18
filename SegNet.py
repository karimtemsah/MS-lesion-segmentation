# Copyright 2017 Christoph Baur <c.baur@tum.de>
# ==============================================================================


import sys
import os
import matplotlib.pyplot
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from MSSEG2008 import *
from model.inference import model
from model.evaluation import loss_calc, evaluation
from model.helpers import *


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
config['batchsize'] = 40
config['learningrate'] = 0.01
config['numEpochs'] = 10
tf.reset_default_graph()

# Define placeholders
inputs = {}
inputs['data'] = tf.placeholder(tf.float32, [config['batchsize'], 128, 128, 1])
inputs['labels'] = tf.placeholder(tf.int32, [config['batchsize'], 128, 128, 1])
inputs['phase'] = tf.placeholder(tf.bool)

# Define a dictionary for storing curves
curves = {}
curves['training'] = []
curves['validation'] = []

logits, probabilities = model(inputs['data'])
printNumberOfTrainableParams()
show_all_variables()
cross_entropy = loss_calc(logits, inputs['labels'])
#correct_prediction = tf.equal(logits, inputs['labels'])
#accuracy_operation = tf.cast(correct_prediction, tf.float32)

train_step = tf.train.GradientDescentOptimizer(config['learningrate']).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

numTrainSamples = dataset.numBatches(config['batchsize'], set='TRAIN')
numValSamples = dataset.numBatches(config['batchsize'], set='VAL')
numTestSamples = dataset.numBatches(config['batchsize'], set='TEST')
print(numTrainSamples)
print(numValSamples)
print(numTestSamples)

for e in range(config['numEpochs']):
    avg_loss_in_current_epoch = 0
    for i in range(0, numTrainSamples):
        batch_data, batch_labels, _ = dataset.next_batch(config['batchsize'])
        batch_data = batch_data.reshape((batch_data.shape[0], 128, 128, 1))
        fetches = {
            'optimizer': train_step,
            'loss': cross_entropy
        }
        results = sess.run(fetches, feed_dict={inputs['data']: batch_data, inputs['labels']: batch_labels})
        avg_loss_in_current_epoch += results['loss']
        if i % 100 == 0:
            print("...")
    avg_loss_in_current_epoch = avg_loss_in_current_epoch / numTrainSamples
    print("Train: ", avg_loss_in_current_epoch)
    curves['training'] += [avg_loss_in_current_epoch]

    for i in range(0, numValSamples):
        # Use Matplotlib to visualize the loss on the training and validation set
        batch_data, batch_labels, _ = dataset.next_batch(config['batchsize'], set='VAL')
        batch_data = batch_data.reshape((batch_data.shape[0], 128, 128, 1))
        batch_labels = batch_labels.reshape((batch_labels.shape[0], 128, 128, 1))
        print(batch_data.shape)
        print(batch_labels.shape)
        fetches = {
            'loss': cross_entropy
        }
        results = sess.run(fetches, feed_dict={inputs['data']: batch_data, inputs['labels']: batch_labels})
        avg_loss_in_current_epoch += results['loss']
    avg_loss_in_current_epoch = avg_loss_in_current_epoch / numValSamples
    curves['validation'] += [avg_loss_in_current_epoch]
    print("VAL: ", avg_loss_in_current_epoch)
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
