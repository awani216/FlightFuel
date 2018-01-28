from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import random
from sklearn import preprocessing



with open(r"../data_Reduction_4.pickle", 'rb') as fl:
    data = pickle.load(fl)
    train_dataset = data['train_dataset']
    train_labels = data['train_labels']
    test_dataset = data['test_dataset']
    attributes =  data['attributes']

train_labels = np.rint(train_labels/10)
print(train_labels)
train_dataset =  preprocessing.scale(train_dataset)
ran = np.random.permutation(len(train_dataset))
train_dataset = train_dataset[ran]
train_labels = train_labels[ran]
train_labels = (np.arange(1260) == train_labels[:,None]).astype(np.float32)

graph =  tf.Graph()

valid_dataset = train_dataset[-11000:-1000]
valid_labels = train_labels[-11000:-1000]

test_dataset = train_dataset[-1000:]
test_labels = train_labels[-1000:]

train_dataset = train_dataset[:-11000]
train_labels =  train_labels[:-11000]
num_labels = 1260
batch_size = 128
num_features = len(train_dataset[0])
train_subset = 10000
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_features))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.to_float(tf.constant(valid_dataset), name='ToFloat')
    tf_test_dataset = tf.to_float(tf.constant(test_dataset), name='ToFloat')

    weights = tf.to_float(tf.Variable(tf.truncated_normal([num_features, num_labels])), name='ToFloat')
    biases = tf.Variable(tf.zeros([num_labels]))

    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

num_steps = 16001

def accuracy(predictions, labels):
    return (100.0 * np.sum(abs(np.argmax(predictions, 1) - np.argmax(labels, 1)) <= 20)/ predictions.shape[0])

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 100 == 0):
            print('Loss at step %d: %f' % (step, l))
            print('Training accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
