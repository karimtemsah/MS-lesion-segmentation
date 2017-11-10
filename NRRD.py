# Imports
import nrrd
import numpy as np
import matplotlib.pyplot as plt
import time


# Class for working with a NII file in the context of machine learning
class NRRD:
    VIEW_MAPPING = {'saggital': 0, 'coronal': 1, 'axial': 2}

    def __init__(self, filename):
        self.data, self.info = nrrd.read(filename)

    @property
    def numSaggitalSlices(self):
        return self.data.shape[NRRD.VIEW_MAPPING['saggital']]

    @property
    def numCoronalSlices(self):
        return self.data.shape[NRRD.VIEW_MAPPING['coronal']]

    @property
    def numAxialSlices(self):
        return self.data.shape[NRRD.VIEW_MAPPING['axial']]

    def setViewMapping(self, mapping):
        NRRD.VIEW_MAPPING = mapping

    def shape(self):
        return self.data.shape

    def getAxisIndex(self, axis):
        return NRRD.VIEW_MAPPING[axis]

    def numSlicesAlongAxis(self, axis):
        return self.data.shape[NRRD.VIEW_MAPPING[axis]]

    def normalize(self, method='scaling', lowerpercentile=None, upperpercentile=None):
        # Convert the attribute "data" to float()
        self.data = self.data.astype(np.float32)

        if lowerpercentile is not None:
            qlow = np.percentile(self.data, lowerpercentile)
        if upperpercentile is not None:
            qup = np.percentile(self.data, upperpercentile)

        if lowerpercentile is not None:
            self.data[self.data < qlow] = qlow
        if upperpercentile is not None:
            self.data[self.data > qup] = qup

        if method == 'scaling':
            # Divide "data" by its maximum value
            self.data -= self.data.min()
            self.data = np.multiply(self.data, 1.0 / self.data.max())
        elif method == 'standardization':
            self.data = self.data - np.mean(self.data)
            self.data = self.data / np.std(self.data)

    def getSlice(self, the_slice, axis='axial'):
        indices = [slice(None)] * self.data.ndim
        indices[NRRD.VIEW_MAPPING[axis]] = the_slice
        return self.data[indices]

    def getData(self):
        return self.data

    def setToZero(self):
        self.data.fill(0.0)

    def visualize(self, axis='axial', pause=0.2):
        for i in range(self.data.shape[NRRD.VIEW_MAPPING[axis]]):
            img = self.getSlice(i, axis=axis)
            plt.imshow(img)
            plt.pause(pause)

    def printStats(self):
        print("NRRD Statistics")
        print("Min: %f" % (self.data.min()))
        print('Max: %f' % (self.data.max()))
        print('Mean: %f' % (self.data.mean()))
        print('Saggital Slices: %d' % (self.numSaggitalSlices))
        print('Coronal Slices: %d' % (self.numCoronalSlices))
        print('Axial Slices: %d' % (self.numAxialSlices))
