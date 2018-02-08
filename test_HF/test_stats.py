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

vect_img=np.reshape(tab_img,[400,360*260])

epoch_img=np.ones((50,93600,8),float)
print(np.shape(epoch_img))

epoch_img[:,:,0]=(vect_img[0:50,:])
epoch_img[:,:,1]=(vect_img[50:100,:])
epoch_img[:,:,2]=(vect_img[100:150,:])
epoch_img[:,:,3]=(vect_img[150:200,:])
epoch_img[:,:,4]=(vect_img[200:250,:])
epoch_img[:,:,5]=(vect_img[250:300,:])
epoch_img[:,:,6]=(vect_img[300:350,:])
epoch_img[:,:,7]=(vect_img[350:400,:])

Label=np.load('Label_test.npy');


epoch_label=np.ones((50,2,8),float)
print(np.shape(epoch_label))



epoch_label[:,:,0]=(Label[0:50,:])
epoch_label[:,:,1]=(Label[50:100,:])
epoch_label[:,:,2]=(Label[100:150,:])
epoch_label[:,:,3]=(Label[150:200,:])
epoch_label[:,:,4]=(Label[200:250,:])
epoch_label[:,:,5]=(Label[250:300,:])
epoch_label[:,:,6]=(Label[300:350,:])
epoch_label[:,:,7]=(Label[350:400,:])

print(np.shape(Label))
print(np.shape(tab_img))
#print(np.shape(tab_img[1][:]))


n_nodes_hl1 = 500 
n_nodes_hl2 = 500 
n_nodes_hl3 = 500

n_classes = 2 
batch_size = 50 

# input feature size = 360*260 pixels = 93600 
x = tf.placeholder('float', [None, 93600]) 
y = tf.placeholder('float') 

exec(open("classifHF.py").read())


