from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import cv2

import tensorflow as tf
import numpy as np
import time

img=[]
tab_img=[]
for i in range(1,201):
    for lettre in ['a','b']:
        img.append(str(i)+lettre)
        
img_path='frontalimages_manuallyaligned_part1/'
#print(img_path+img[2]+'.jpg')


for i in range(0,400):
    image=cv2.imread(img_path+img[i]+'.jpg',0)
    tab_img.append(image)

#Label=[]

Label=np.load('Label_test.npy');

print(np.shape(Label))
print(np.shape(tab_img))
vect_img=[]
print(np.shape(tab_img[1][:]))

vect_img=np.reshape(tab_img,[400,360*260])
 
print(np.shape(vect_img))

print(np.shape(vect_img[1]))
#cv2.imshow('image',vect_img[399])
#cv2.waitKey(0)
#cv2.destroyAllWindows()

tict = time.time()



x = tf.placeholder(tf.float32, [None, 260*360])

W = tf.Variable(tf.zeros([260*360, 2]))
b = tf.Variable(tf.zeros([2]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None,2])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(tf.matmul(x, W) + b,
#  labels_placeholder))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()


tf.global_variables_initializer().run()

with tf.device('/gpu:1'):
    for _ in range(1000):
      sess.run(train_step, feed_dict={x: vect_img[1:200], y_: Label[1:200]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

elapsed = time.time() - tict
print(elapsed)
print('\n')
print(sess.run(accuracy, feed_dict={x: vect_img[201:399], y_: Label[201:399]}))


