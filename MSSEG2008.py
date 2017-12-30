from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import csv
import glob
import pickle
import math
import os.path
import matplotlib.pyplot
import copy
import scipy.misc
from six.moves import xrange  # pylint: disable=redefined-builtin

from NRRD import *
from image_utils import crop, crop_center
from tfrecord_utils import *
from scipy import misc
from random import randint
from skimage import measure

matplotlib.pyplot.ion()


# DATASET CLASS
class MSSEG2008(object):
    # Class Variables
    PROTOCOL_MAPPINGS = ['FLAIR', 'T1', 'T2']
    SET_TYPES = ['TRAIN', 'VAL', 'TEST']

    class Options(object):
        def __init__(self):
            self.dir = os.path.dirname(os.path.realpath(__file__))
            self.folderTrain = 'train'
            self.folderTest = 'test'
            self.numSamples = -1
            self.partition = {'TRAIN': 0.7, 'VAL': 0.3}
            self.useCrops = False
            self.cropType = 'random'  # random or center
            self.numRandomCropsPerSlice = 5
            self.onlyPatchesWithLesions = False
            self.rotations = 0
            self.cropWidth = 128
            self.cropHeight = 128
            self.cache = False
            self.sliceResolution = None  # format: HxW
            self.addInstanceNoise = False  # Affects only the batch sampling. If True, a tiny bit of noise will be added to every batch
            self.filterProtocol = None  # FLAIR, T1, T2
            self.axis = 'axial'  # saggital, coronal or axial
            self.debug = False
            self.normalizationMethod = 'standardization'

    def __init__(self, options=Options()):
        self.options = options

        if options.cache and os.path.isfile(self.pckl_name()):
            f = open(self.pckl_name(), 'rb')
            tmp = pickle.load(f)
            f.close()
            self._epochs_completed = tmp._epochs_completed
            self._index_in_epoch = tmp._index_in_epoch
            self.patientsSplit = tmp.patientsSplit
            self.patients = tmp.patients
            self._images, self._labels, self._sets = readTFRecord(self.tfrecord_name())
        else:
            # Collect all patients
            self.patients, self.patientsSplit = self._getPatients()

            if not os.path.isfile(self.split_name()):
                f = open(self.split_name(), 'wb')
                pickle.dump(self.patientsSplit, f)
                f.close()
            else:
                f = open(self.split_name(), 'rb')
                self.patientsSplit = pickle.load(f)
                f.close()

            # Iterate over all patients and extract slices
            _images = []
            _labels = []
            _sets = []
            for p, patient in enumerate(self.patients):
                if p in self.patientsSplit['TRAIN']:
                    _set_of_current_patient = MSSEG2008.SET_TYPES.index('TRAIN')
                elif p in self.patientsSplit['VAL']:
                    _set_of_current_patient = MSSEG2008.SET_TYPES.index('VAL')
                elif p in self.patientsSplit['TEST']:
                    continue  # We dont need to crop from testing images

                for pr, protocol in enumerate(MSSEG2008.PROTOCOL_MAPPINGS):
                    if self.options.numSamples > 0 and len(_images) > self.options.numSamples:
                        break
                    if self.options.filterProtocol and protocol not in self.options.filterProtocol:
                        continue

                    nrrd = NRRD(patient[protocol])
                    nrrd.printStats()
                    # nrrd.visualize(axis=self.options.axis)
                    nrrd_groundtruth = NRRD(patient['groundtruth'])
                    nrrd_groundtruth.printStats()

                    # In-place normalize the loaded volume
                    nrrd.normalize(method=options.normalizationMethod)

                    # Iterate over all slices and collect them
                    for s in xrange(0, nrrd.numSlicesAlongAxis(self.options.axis)):
                        if self.options.numSamples > 0 and len(_images) > self.options.numSamples:
                            break

                        slice_data = nrrd.getSlice(s, self.options.axis)
                        slice_seg = nrrd_groundtruth.getSlice(s, self.options.axis)

                        if self.options.sliceResolution is not None:
                            slice_data = scipy.ndimage.interpolation.zoom(slice_data,
                                                                          float(self.options.sliceResolution) / float(
                                                                              slice_data.shape))
                            slice_seg = scipy.ndimage.interpolation.zoom(slice_seg,
                                                                         float(self.options.sliceResolution) / float(
                                                                             slice_data.shape), mode="nearest")

                        for angle in self.options.rotations:
                            if self.options.numSamples > 0 and len(_images) > self.options.numSamples:
                                break

                            if angle != 0:
                                slice_data_rotated = scipy.ndimage.interpolation.rotate(slice_data, angle,
                                                                                        reshape=False)
                                slice_seg_rotated = scipy.ndimage.interpolation.rotate(slice_seg, angle, reshape=False,
                                                                                       mode='nearest')
                            else:
                                slice_data_rotated = slice_data
                                slice_seg_rotated = slice_seg

                            # Either collect crops
                            if self.options.useCrops:
                                if self.options.cropType == 'random':
                                    rx = numpy.random.randint(0, high=(
                                            slice_data_rotated.shape[1] - self.options.cropWidth),
                                                              size=self.options.numRandomCropsPerSlice)
                                    ry = numpy.random.randint(0, high=(
                                            slice_data_rotated.shape[0] - self.options.cropHeight),
                                                              size=self.options.numRandomCropsPerSlice)
                                    for r in range(self.options.numRandomCropsPerSlice):
                                        if self.options.numSamples > 0 and len(_images) > self.options.numSamples:
                                            break

                                        slice_data_cropped = crop(slice_data_rotated, ry[r], rx[r],
                                                                  self.options.cropHeight, self.options.cropWidth)
                                        slice_seg_cropped = crop(slice_seg_rotated, ry[r], rx[r],
                                                                 self.options.cropHeight, self.options.cropWidth)

                                        if self.options.onlyPatchesWithLesions and numpy.max(
                                                slice_seg_cropped) == 0:  # This means there is no lesion inside this patch
                                            continue

                                        _images.append(slice_data_cropped)
                                        _labels.append(slice_seg_cropped)
                                        _sets.append(_set_of_current_patient)
                                elif self.options.cropType == 'center':
                                    slice_data_cropped = crop_center(slice_data_rotated, self.options.cropWidth,
                                                                     self.options.cropHeight)
                                    slice_seg_cropped = crop_center(slice_seg_rotated, self.options.cropWidth,
                                                                    self.options.cropHeight)

                                    if not self.options.onlyPatchesWithLesions or numpy.max(slice_seg_cropped) > 0:
                                        _images.append(slice_data_cropped)
                                        _labels.append(slice_seg_cropped)
                                        _sets.append(_set_of_current_patient)
                            # Or whole slices
                            else:
                                if not self.options.onlyPatchesWithLesions or numpy.max(slice_seg_rotated) > 0:
                                    _images.append(slice_data_rotated)
                                    _labels.append(slice_seg_rotated)
                                    _sets.append(_set_of_current_patient)

            self._images = numpy.array(_images).astype(numpy.float32)
            self._labels = numpy.array(_labels).astype(numpy.float32)
            if self._images.ndim < 4:
                self._images = numpy.expand_dims(self._images, 3)
            self._sets = numpy.array(_sets).astype(numpy.int32)
            self._epochs_completed = [0, 0]
            self._index_in_epoch = [0, 0]

            if self.options.cache:
                writeTFRecord(self._images, self._labels, self._sets, self.tfrecord_name())
                tmp = copy.copy(self)
                tmp._images = None
                tmp._labels = None
                tmp._sets = None
                f = open(self.pckl_name(), 'wb')
                pickle.dump(tmp, f)
                f.close()

    # Hidden helper function, not supposed to be called from outside!
    def _getPatients(self):
        patients = []
        patientsSplit = {}

        # Get all files that can be used for training and validation
        patients_train = [f.name for f in os.scandir(os.path.join(self.dir(), self.options.folderTrain)) if f.is_dir()]
        for p, pname in enumerate(patients_train):
            patient = {}
            patient['name'] = pname
            numLoadedVols = 0
            for pr, protocol in enumerate(MSSEG2008.PROTOCOL_MAPPINGS):
                try:
                    patient[protocol] = os.path.join(self.dir(), self.options.folderTrain, pname,
                                                     pname + '_' + protocol + '.nhdr')
                    numLoadedVols += 1
                except:
                    print('MSSEG2008: Failed to open file ' + pname + '_' + protocol + '.nhdr')
            try:
                patient['groundtruth'] = os.path.join(self.dir(), self.options.folderTrain, pname,
                                                      pname + '_lesion.nhdr')
            except:
                print('MSSEG2008: Failed to open file ' + pname + '_lesion.nhdr')

            # Append to the list of all patients
            if numLoadedVols > 0:
                patients.append(patient)

        # Determine Train, Val & Test set based on patients
        _numPatients = len(patients)
        _numTrain = math.floor(self.options.partition['TRAIN'] * _numPatients)
        _numVal = math.floor(self.options.partition['VAL'] * _numPatients)
        _ridx = numpy.random.permutation(_numPatients)
        patientsSplit['TRAIN'] = _ridx[0:_numTrain]
        patientsSplit['VAL'] = _ridx[_numTrain:_numTrain + _numVal]

        # Get all files that are meant to be used for testing
        patients_test = [f.name for f in os.scandir(os.path.join(self.dir(), self.options.folderTest)) if f.is_dir()]
        for p, pname in enumerate(patients_test):
            patient = {}
            patient['name'] = pname
            numLoadedVols = 0
            for pr, protocol in enumerate(MSSEG2008.PROTOCOL_MAPPINGS):
                try:
                    patient[protocol] = os.path.join(self.dir(), self.options.folderTest, pname,
                                                     pname + '_' + protocol + '.nhdr')
                    numLoadedVols += 1
                except:
                    print('MSSEG2008: Failed to open file ' + pname + '_' + protocol + '.nhdr')

            # Append to the list of all patients
            if numLoadedVols > 0:
                patients.append(patient)

        # Determine indices for testing patients
        patientsSplit['TEST'] = list(range(_numPatients, len(patients)))

        return patients, patientsSplit

    # Returns the indices of patients which belong to either TRAIN, VAL or TEST. Your choice
    def getPatientsinSet(self, split='TRAIN'):
        return self.patientsSplit[split]

    @property
    def images(self):
        return self._images

    def getImage(self, i):
        return self._images[i, :, :, :]

    def getLabel(self, i):
        return self._labels[i, :, :, :]

    def getPatient(self, i):
        return self.patients[i]

    @property
    def labels(self):
        return self._labels

    @property
    def sets(self):
        return self._sets

    @property
    def meta(self):
        return self._meta

    @property
    def num_examples(self):
        return self._images.shape[0]

    @property
    def width(self):
        return self._images.shape[2]

    @property
    def height(self):
        return self._images.shape[1]

    @property
    def num_channels(self):
        return self._images.shape[3]

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def name(self):
        str = "MSSEG2008"
        if self.options.numSamples > 0:
            str += '_n{}'.format(self.options.numSamples)
        str += "_p{}-{}".format(self.options.partition['TRAIN'], self.options.partition['VAL'])
        if self.options.useCrops:
            str += "_{}crops{}x{}".format(self.options.cropType, self.options.cropWidth, self.options.cropHeight)
            if self.options.cropType == "random":
                str += "_{}cropsPerSlice".format(self.options.numRandomCropsPerSlice)
        if self.options.sliceResolution is not None:
            str += "_res{}x{}".format(self.options.sliceResolution[0], self.options.sliceResolution[1])
        return str

    def split_name(self):
        return os.path.join(self.dir(),
                            'split-{}-{}.pckl'.format(self.options.partition['TRAIN'], self.options.partition['VAL']))

    def pckl_name(self):
        return os.path.join(self.dir(), self.name() + ".pckl")

    def tfrecord_name(self):
        return os.path.join(self.dir(), self.name() + ".tfrecord")

    def dir(self):
        return self.options.dir

    def exportSlices(self, dir):
        for i in range(self.num_examples):
            scipy.misc.imsave(os.path.join(dir, '{}.png'.format(i)), np.squeeze(self.getImage(i)))

    def visualize(self, pause=1):
        f, (ax1, ax2) = matplotlib.pyplot.subplots(1, 2)
        images_tmp, labels_tmp, _ = self.next_batch(10)
        for i in range(images_tmp.shape[0]):
            img = numpy.squeeze(images_tmp[i])
            lbl = numpy.squeeze(labels_tmp[i])
            ax1.imshow(img)
            ax1.set_title('Patch')
            ax2.imshow(lbl)
            ax2.set_title('Groundtruth')
            #matplotlib.pyplot.pause(pause)

    def printStats(self):
        print("Dataset Statistics")
        print("Min: %f" % (self._images.min()))
        print('Max: %f' % (self._images.max()))
        print('Mean: %f' % (self._images.mean()))
        print('Number of Images: %d' % (self.num_examples))
        print('Number of Channels: %d' % (self.num_channels))
        print('Width: %d' % (self.width))
        print('Height: %d' % (self.height))

    def numBatches(self, batchsize, set='TRAIN'):
        _setIdx = MSSEG2008.SET_TYPES.index(set)
        images_in_set = numpy.where(self._sets == _setIdx)[0]
        return len(images_in_set) // batchsize

    def next_batch(self, batch_size, shuffle=True, set='TRAIN'):
        """Return the next `batch_size` examples from this data set."""
        _setIdx = MSSEG2008.SET_TYPES.index(set)
        images_in_set = numpy.where(self._sets == _setIdx)[0]
        samples_in_set = len(images_in_set)

        start = self._index_in_epoch[_setIdx]
        # Shuffle for the first epoch
        if self._epochs_completed[_setIdx] == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(samples_in_set)
            numpy.random.shuffle(perm0)
            self._images[images_in_set] = self.images[images_in_set[perm0]]
            self._labels[images_in_set] = self.labels[images_in_set[perm0]]
            self._sets[images_in_set] = self.sets[images_in_set[perm0]]

        # Go to the next epoch
        if start + batch_size > samples_in_set:
            # Finished epoch
            self._epochs_completed[_setIdx] += 1

            # Get the rest examples in this epoch
            rest_num_examples = samples_in_set - start
            images_rest_part = self._images[images_in_set[start:samples_in_set]]
            labels_rest_part = self._labels[images_in_set[start:samples_in_set]]

            # Shuffle the data
            if shuffle:
                perm = numpy.arange(samples_in_set)
                numpy.random.shuffle(perm)
                self._images[images_in_set] = self.images[images_in_set[perm]]
                self._labels[images_in_set] = self.labels[images_in_set[perm]]
                self._sets[images_in_set] = self.sets[images_in_set[perm]]

            # Start next epoch
            start = 0
            self._index_in_epoch[_setIdx] = batch_size - rest_num_examples
            end = self._index_in_epoch[_setIdx]
            images_new_part = self._images[images_in_set[start:end]]
            labels_new_part = self._labels[images_in_set[start:end]]

            images_tmp = numpy.concatenate((images_rest_part, images_new_part), axis=0)
            labels_tmp = numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch[_setIdx] += batch_size
            end = self._index_in_epoch[_setIdx]
            images_tmp = self._images[images_in_set[start:end]]
            labels_tmp = self._labels[images_in_set[start:end]]

        if self.options.addInstanceNoise:
            noise = numpy.random.normal(0, 0.01, images_tmp.shape)
            images_tmp += noise

        # Check the batch
        assert images_tmp.size, "The batch is empty!"
        assert labels_tmp.size, "The labels of the current batch are empty!"

        return images_tmp, labels_tmp, None
