
import tensorflow as tf


def loss_calc(logits, labels):

    class_inc_bg = 2

    labels = labels[...,0]
    class_weights = tf.constant([[10.0/90, 10.0]])

    onehot_labels = tf.one_hot(labels, class_inc_bg)
    weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)

    unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits)

    weighted_losses = unweighted_losses * weights

    loss = tf.reduce_mean(weighted_losses)

    tf.summary.scalar('loss', loss)
    return loss


def evaluation(logits, labels):
    labels = labels[..., 0]

    correct_prediction = tf.equal(tf.argmax(logits, 3), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy


def conf_matrix(prediction, label, num_classes=2):
    with tf.name_scope("test"):
        # Compute a per-batch confusion
        labelff = label[..., 0]
        labelff = tf.reshape(labelff, (-1,))
        predictionff = tf.argmax(prediction, 3)
        predictionff = tf.reshape(predictionff, (-1,))
        batch_confusion = tf.confusion_matrix(labelff, predictionff,
                                              num_classes=num_classes)
        # Create an accumulator variable to hold the counts
        confusion = tf.Variable(tf.zeros([num_classes,num_classes],
                                          dtype=tf.int32),
                                 name='confusion')
        # Create the update op for doing a "+=" accumulation on the batch
        confusion_update = confusion.assign(confusion + batch_confusion)
        # Cast counts to float so tf.summary.image renormalizes to [0,255]
        confusion_image = tf.reshape(tf.cast(confusion_update, tf.float32),
                                     [1, num_classes, num_classes, 1])

        tf.summary.image('confusion', confusion_image)
    return confusion_update
