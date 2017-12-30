
# Basic imports
import sys
import os
import matplotlib.pyplot
import numpy as np


# Further imports
from MSSEG2008 import *


# Load and prepare dataset
dataset_options = MSSEG2008.Options()
dataset_options.debug = True
dataset_options.dir = '/media/data/MSSEG2008/'
dataset_options.useCrops = True
dataset_options.cropType = 'random'
dataset_options.cropWidth = 128
dataset_options.cropHeight = 128
dataset_options.numRandomCropsPerSlice = 20 # Not needed when doing center crops
dataset_options.onlyPatchesWithLesions = True
dataset_options.rotations = range(-2,2)
dataset_options.partition = {'TRAIN': 0.6, 'VAL': 0.4}
dataset_options.sliceResolution = None # We do not need that since we extract crops!
dataset_options.cache = True
dataset_options.numSamples = -1
dataset_options.addInstanceNoise = False
dataset_options.axis = 'axial'
dataset_options.filterProtocol = ['FLAIR']
dataset_options.normalizationMethod = 'standardization'

# Center Crops of healthy control images: training, validation and testing patients
dataset = MSSEG2008(dataset_options)
#dataset.visualize()

# Testing next_batch function for train & val
batchsize = 40
numEpochs = 5
for epoch in xrange(numEpochs):
    numBatches = dataset.numBatches(batchsize, set='TRAIN')
    for idx in xrange(0, numBatches):
        batch_real, _, _ = dataset.next_batch(batchsize, set='TRAIN')
    print('Done with training epoch ({} batches)'.format(numBatches))

    numBatches = dataset.numBatches(batchsize, set='VAL')
    for idx in xrange(0, numBatches):
        batch_real, _, _ = dataset.next_batch(batchsize, set='VAL')
    print('Done with validation ({} batches)'.format(numBatches))



# How to use the dataset during validation (or testing)
indices_of_testing_patients = dataset.getPatientsinSet(split='VAL')
for i in indices_of_testing_patients:
    patient = dataset.getPatient(i)

    nrrd = NRRD(patient[dataset_options.filterProtocol[0]])
    nrrd.printStats()
    #nrrd.visualize(axis=dataset_options.axis)

    nrrd_groundtruth = NRRD(patient['groundtruth'])
    nrrd_groundtruth.printStats()
    #nrrd_groundtruth.visualize(axis=dataset_options.axis)

    # In-place normalize the loaded volume
    nrrd.normalize(method=dataset_options.normalizationMethod)

    # Now go slice by slice over the entire volume and segment every slice.
    nrrd_prediction = np.zeros(nrrd_groundtruth.shape())
    for i in range(nrrd.numSlicesAlongAxis(dataset_options.axis)):
        _slice = nrrd.getSlice(i, dataset_options.axis)
        #prediction = PREDICT USING YOUR MODEL
        #nrrd_prediction[:,:,i] = prediction

    # Threshold the volume
    # Compute metrics between predicted volume and groundtruth
    # Save prediction as a NRRD file again.
