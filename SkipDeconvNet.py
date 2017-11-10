# Copyright 2017 Christoph Baur <c.baur@tum.de>
# ==============================================================================


# Basic imports
import sys
import os
import matplotlib.pyplot
import numpy as np
import tensorflow as tf


def model(input, scope="SkipDeconvNet", reuse=True):
    # TODO...