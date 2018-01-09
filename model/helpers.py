import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import numpy as np

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


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


def add_output_images(images, logits, labels, max_outputs=5):

    tf.summary.image('input', images, max_outputs=max_outputs)

    output_image_bw = images[..., 0]

    labels1 = tf.cast(labels[...,0], tf.float32)

    input_labels_image_r = labels1 + (output_image_bw * (1-labels1))
    input_labels_image = tf.stack([input_labels_image_r, output_image_bw, output_image_bw], axis=3)
    tf.summary.image('input_labels_mixed', input_labels_image, max_outputs=5)

    classification1 = tf.nn.softmax(logits = logits, dim=-1)[...,1]

    output_labels_image_r = classification1 + (output_image_bw * (1-classification1))
    output_labels_image = tf.stack([output_labels_image_r, output_image_bw, output_image_bw], axis=3)
    tf.summary.image('output_labels_mixed', output_labels_image, max_outputs=5)

    return
