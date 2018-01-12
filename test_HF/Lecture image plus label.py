from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import cv2

import tensorflow as tf
import numpy as np

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

Label=[]

fichier = open("Label.txt", "r")

while 1:
    txt = fichier.readline()
    if txt =='':
        break
    else:
        Label.append(txt[0])
        
fichier.close()

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


x = tf.placeholder(tf.float32, [None, 260*360])

W = tf.Variable(tf.zeros([260*360, 2]))
b = tf.Variable(tf.zeros([2]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, ])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(1000):
  sess.run(train_step, feed_dict={x: vect_img[1:300], y_: Label[1:300]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: vect_img[301:399], y_: Label[301:399]}))


