import pickle
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix


def threshold(predictions, labels):
    labels = tf.cast(tf.reshape(labels, (-1, 1)), tf.float32)
    #labels = tf.reshape(tf.one_hot(label_flat, depth=2), (-1, 2))
    predictions = tf.reshape(predictions, (-1, 2))
    predictions = tf.nn.softmax(predictions)
    predictions = predictions[:,1]
    predictions = tf.reshape(predictions, (-1, 1))
    return labels, predictions


def best_threshold(predictions, labels):
    th = np.arange(0.01, 1.01, 0.01)
    th = th.tolist()
    best_conf = np.array([[0, 0], [0, 0]])
    best_th = 0
    best_F1 = 0
    for i in th:
        print(i)
        modified_pred = np.zeros(predictions.shape[0])
        modified_pred[np.where(predictions >= i)[0]] = 1
        conf = confusion_matrix(labels, modified_pred)

        f1 = 2 * conf[1][1] / (2 * conf[1][1] + conf[0][1] + conf[1][0])
        if f1 > best_F1:
            best_th = i
            best_F1 = f1
            best_conf = conf
            print(f1, best_conf)
    return best_conf, best_th, best_F1


preds = pickle.load(open('../preds.pkl', 'rb'))
lbls = pickle.load(open('../lbls.pkl', 'rb'))
print(best_threshold(preds, lbls))
