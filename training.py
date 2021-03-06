# Copyright 2017 Christoph Baur <c.baur@tum.de>
# ==============================================================================


import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from MSSEG2008 import *
from model.inference import model, model_droput
from model.evaluation import loss_calc, evaluation, conf_matrix, loss
from model.helpers import *



dataset_options = MSSEG2008.Options()
dataset_options.debug = True
dataset_options.dir = '/media/data/MSSEG2008/'
dataset_options.useCrops = True
dataset_options.cropType = 'random'
dataset_options.cropWidth = 128
dataset_options.cropHeight = 128
dataset_options.numRandomCropsPerSlice = 20 # Not needed when doing center crops
dataset_options.onlyPatchesWithLesions = False
dataset_options.rotations = range(-2,2)
dataset_options.partition = {'TRAIN': 0.7, 'VAL': 0.3}
dataset_options.sliceResolution = None # We do not need that since we extract crops!
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
config['numEpochs'] = 50
tf.reset_default_graph()

# Define placeholders
inputs = {}
inputs['data'] = tf.placeholder(tf.float32, [config['batchsize'], 128, 128, 1], name="input")
inputs['labels'] = tf.placeholder(tf.int64, [config['batchsize'], 128, 128, 1], name="labels")
inputs['phase'] = tf.placeholder(bool)

# Define a dictionary for storing curves
curves = {}
curves['training'] = []
curves['validation'] = []

logits = model_droput(inputs['data'], inputs['phase'])
add_output_images(images=inputs['data'], logits=logits, labels=inputs['labels'])
printNumberOfTrainableParams()
show_all_variables()
cross_entropy = loss(logits, inputs['labels'])
#correct_prediction = tf.equal(logits, inputs['labels'])
accuracy_operation = evaluation(logits, inputs['labels'])
confusion = conf_matrix(logits, inputs['labels'])

train_step = tf.train.GradientDescentOptimizer(config['learningrate']).minimize(cross_entropy)
summary = tf.summary.merge_all()
sess = tf.InteractiveSession()

train_writer = tf.summary.FileWriter('run5', sess.graph)
tf.global_variables_initializer().run()
numTrainSamples = dataset.numBatches(config['batchsize'], set='TRAIN')
numValSamples = dataset.numBatches(config['batchsize'], set='VAL')
#numTestSamples = dataset.numBatches(config['batchsize'], set='TEST')
print(numTrainSamples)
print(numValSamples)
#print(numTestSamples)
saver = tf.train.Saver()
last_conf = np.array([[0,0], [0,0]])
dice = []
epocs = []
for e in range(config['numEpochs']):
    avg_loss_in_current_epoch = 0
    for i in range(0, numTrainSamples):
        batch_data, batch_labels, _ = dataset.next_batch(config['batchsize'])
        batch_data = batch_data.reshape((batch_data.shape[0], 128, 128, 1))
        fetches = {
            'optimizer': train_step,
            'loss': cross_entropy
        }
        results = sess.run(fetches, feed_dict={inputs['data']: batch_data, inputs['labels']: batch_labels, inputs['phase']:True})
        avg_loss_in_current_epoch += results['loss']
        if i % 100 == 0:
            #train_writer.add_summary(results['loss'], i)
            print(i)
    avg_loss_in_current_epoch = avg_loss_in_current_epoch / numTrainSamples
    print("Train: ", avg_loss_in_current_epoch)
    curves['training'] += [avg_loss_in_current_epoch]
    #train_writer.add_summary(results['summary'], e)
    if e % 5 == 0:
        saver.save(sess, 'segnet_train_model_5', global_step=5)
    accumulated_predictions = np.array([])
    for i in range(0, numValSamples):
        # Use Matplotlib to visualize the loss on the training and validation set
        batch_data, batch_labels, _ = dataset.next_batch(config['batchsize'], set='VAL')
        batch_data = batch_data.reshape((batch_data.shape[0], 128, 128, 1))
        batch_labels = batch_labels.reshape((batch_labels.shape[0], 128, 128, 1))
        fetches = {
            'loss': cross_entropy,
            'accuracy': accuracy_operation,
            'summary': summary,
            'confusion': confusion
        }

        results = sess.run(fetches, feed_dict={inputs['data']: batch_data, inputs['labels']: batch_labels, inputs['phase']:False})
        avg_loss_in_current_epoch += results['loss']
        if i == 0:
            accumulated_predictions = results['accuracy']
        else:
            accumulated_predictions = np.append(accumulated_predictions, results['accuracy'])
    avg_loss_in_current_epoch = avg_loss_in_current_epoch / numValSamples
    curves['validation'] += [avg_loss_in_current_epoch]
    print("VAL: ", avg_loss_in_current_epoch)
    accuracy = np.mean(accumulated_predictions)
    train_writer.add_summary(results['summary'], e)
    print("Accuracy: ", accuracy)
    temp = results['confusion'] - last_conf
    np.savetxt("conf_matrix.txt", temp)
    print("Confusion Matrix: ", temp)
    dice_score = (2 * temp[1][1])/(2*temp[1][1] + temp[0][1] + temp[1][0])
    print("Dice score:", dice_score)
    dice.append(dice_score)
    epocs.append(e)
    last_conf = results['confusion']
    print('Done with epoch %d' % (e))
    visualizeCurves(curves)

plt.figure("dice")
plt.plot(epocs, dice)
plt.xlabel("epochs")
plt.ylabel("Dice score")
plt.savefig("dice_score.png")


"""
#with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('my_test_model-5.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
"""