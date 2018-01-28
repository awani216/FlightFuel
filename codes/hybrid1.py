import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn import tree
from six.moves import cPickle as pickle
import random
import os, shutil

with open(r"../random_forest.pickle", 'rb') as f1:
    data = pickle.load(f1)
    train_dataset = data['train_dataset']
    train_labels = data['train_labels']
    test_dataset = data['test_dataset']
    attributes = data['attributes']

train_labels = np.array(train_labels, dtype = int)

ran = np.random.permutation(len(train_dataset))
train_dataset = train_dataset[ran]
train_labels = train_labels[ran]

mean = np.mean(train_dataset, axis = 0)
std = np.std(train_dataset, axis = 0)

train_dataset = (train_dataset - mean)/std

@jit
def fun(dataset, label):
    clf_nm = ["ExtraTreeRegressor", "RandomForestRegressor", "KNeighborsRegressor", "DecisionTreeRegressor"]
    clf_arr = [ExtraTreesRegressor(), RandomForestRegressor(), KNeighborsRegressor(), tree.DecisionTreeRegressor() ]
    test_dataset = dataset[-11000:]
    test_labels = label[-11000:]
    dataset = dataset[:-11000]
    labels = label[:-11000]
    result = np.empty(shape=(len(test_dataset),len(clf_arr)))
    for i in range(len(clf_arr)):
        ct = 0
        stime = time.time()
        result_temp = np.empty(shape=(len(test_dataset),7))
        for j in range(7):
            clf = clf_arr[i]
            ran = random.sample(range(len(dataset)),50000)
            train_dataset = dataset[ran]
            train_label = label[ran]
            clf.fit(train_dataset,train_label)
            res = clf.predict(test_dataset)
            result_temp [:,j] = res
        for j in range(len(test_dataset)):
            temp = result_temp[j]
            temp.sort()
            result[j,i] = temp[3]
            if abs(result[j,i] - test_labels[j]) <=120:
                ct+=1
        print("Accuracy for ",clf_nm[i]," is ", (ct/len(test_dataset))*100," and time taken ",time.time()-stime) 
    return result,test_labels
@jit
def fun1(dataset,labels):
    clf_nm = ["ExtraTreeRegressor", "RandomForestRegressor", "KNeighborsRegressor", "DecisionTreeRegressor"]
    clf_arr = [ExtraTreesRegressor(), RandomForestRegressor(), KNeighborsRegressor(), tree.DecisionTreeRegressor() ]
    test_dataset = dataset[-1000:]
    test_labels = labels[-1000:]
    dataset = dataset[:-1000]
    labels = labels[:-1000] 

    for i in range(len(clf_arr)):
        stime = time.time()
        ct = 0
        clf = clf_arr[i]
        clf.fit(dataset,labels)
        result = clf.predict(test_dataset)
        for j in range(len(test_dataset)):
            if abs(result[j]-test_labels[j]) <= 120 :
                ct+=1
        print("Accuracy for ",clf_nm[i]," is ", (ct/len(test_dataset))*100," and time taken ",time.time()-stime) 

@jit
def funten(dataset,labels):
    labels = np.rint(labels/10)
    ran = np.random.permutation(len(dataset))
    dataset = dataset[ran]
    labels = labels[ran]
    train_labels = (np.arange(1260) == train_labels[:,None]).astype(np.float32)
    
    train_dataset = dataset[0:8000]
    train_labels = labels[:8000]

    valid_dataset = dataset[-2000:]
    valid_labels = labels[-2000:]

    test_dataset = dataset[-3000:-2000]
    test_labels = labels[-3000:-2000] 
    graph = tf.Graph()

    num_labels = 1260
    batch_size = 128
    num_features = len(test_dataset[0])

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

    num_steps = 8001
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


rslt,labs = fun(train_dataset,train_labels)
fun1(rslt,labs)
funten(rslt,labs)


