import tensorflow as tf
import numpy as np


"""
def conv_layer(input, filters, kernel_size=3):
    return tf.layers.conv2d(
            inputs=input,
            filters=filters,
            kernel_size=[kernel_size, kernel_size],
            padding="same",
            activation=tf.nn.relu
            )
"""
def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale*tf.where(x >= 0.0, x, alpha*tf.nn.elu(x))


def conv_layer(net, filters, kernel_size=[3,3], activation=True):
    kernal_units = kernel_size[0] * kernel_size[1] * net.shape.as_list()[-1]
    net = tf.layers.conv2d(net, filters, kernel_size,
                           padding='same',
                           activation=None,
                           use_bias=True,
                           bias_initializer=tf.zeros_initializer(),
                           kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=np.sqrt(1/kernal_units))
                           )

    if activation:
        net = selu(net)

    return net


def maxpool_layer(input):
    return tf.nn.max_pool_with_argmax(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


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
        deconv_13 = conv_layer(unpool_5, 512)
        deconv_12 = conv_layer(deconv_13, 512)
        deconv_11 = conv_layer(deconv_12, 512)

        # forth layer
        unpool_4 = unpool_layer(deconv_11, indicies_4)
        deconv_10 = conv_layer(unpool_4, 512)
        deconv_9 = conv_layer(deconv_10, 512)
        deconv_8 = conv_layer(deconv_9, 256)

        # third layer
        unpool_3 = unpool_layer(deconv_8, indicies_3)
        deconv_7 = conv_layer(unpool_3, 256)
        deconv_6 = conv_layer(deconv_7, 256)
        deconv_5 = conv_layer(deconv_6, 128)

        # second layer
        unpool_2 = unpool_layer(deconv_5, indicies_2)
        deconv_4 = conv_layer(unpool_2, 128)
        deconv_3 = conv_layer(deconv_4, 64)

        # first layer
        unpool_1 = unpool_layer(deconv_3, indicies_1)
        deconv_2 = conv_layer(unpool_1, 64)
        #deconv_1 = deconv_layer(deconv_2, 64)

        # Classification
        logits = conv_layer(deconv_2, 2, activation=False)
        softmax = tf.nn.softmax(logits)
        return logits, softmax