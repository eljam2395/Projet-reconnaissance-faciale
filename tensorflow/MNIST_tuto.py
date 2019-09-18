from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import cv2

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



##img=[]
##vect_img=[]
##for i in range(1,201):
##    for lettre in ['a','b']:
##        img.append(str(i)+lettre)
##        
##img_path='F:/Documents/Iris 3A/Projet 3A/Git/Projet-reconnaissance-faciale/test_HF/frontalimages_manuallyaligned_part1/'
##print(img_path+img[2]+'.jpg')
##
##for i in range(1,201):
##    image=cv2.imread(img_path+img[i]+'.jpg',0)
##    vect_img.append(image)
    

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
