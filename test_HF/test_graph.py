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

Label=np.load('Label.npy');

print(np.shape(Label))
print(np.shape(tab_img))
print(np.shape(tab_img[1][:]))

#vect_img=np.reshape(tab_img,[400,360*260])
 



tict = time.time()

# Initialize placeholders 
x=tf.placeholder(dtype = tf.float32, shape = [None, 360, 260])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer 
logits = tf.contrib.layers.fully_connected(images_flat, 10, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y,logits = logits))
# Define an optimizer 
train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.set_random_seed(1234)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
for i in range(20):
    _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: tab_img[i:i+19], y: Label[i:i+19]})
    _, loss_value = sess.run([train_op, loss], feed_dict={x: tab_img[i:i+19], y: Label[i:i+19]})

    print("accuracy_val: ", accuracy_val)
    print("loss_value: ", loss_value)
     


elapsed = time.time() - tict
##print(elapsed)
##print('\n')
##print(accuracy_val)
##print((sess.run([correct_pred], feed_dict={x: tab_img})[0]))
#print(sess.run(accuracy,feed_dict={x: vect_img[350:399], y: Label[350:399]}))


