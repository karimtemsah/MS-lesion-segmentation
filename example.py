# Copyright 2017 Christoph Baur <c.baur@tum.de>
# ==============================================================================

"""Example script on how to use the MSSEG2008 dataset class"""


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
dataset_options.numRandomCropsPerSlice = 10 # Not needed when doing center crops
dataset_options.onlyPatchesWithLesions = True
dataset_options.rotations = range(-5,5)
dataset_options.partition = {'TRAIN': 0.6, 'VAL': 0.4}
dataset_options.sliceResolution = None
dataset_options.cache = True
dataset_options.numSamples = 10
dataset_options.addInstanceNoise = False
dataset_options.axis = 'axial'
dataset_options.filterProtocol = ['FLAIR']
dataset_options.normalizationMethod = 'standardization'

# Center Crops of healthy control images: training, validation and testing patients
dataset = MSSEG2008(dataset_options)
dataset.visualize()