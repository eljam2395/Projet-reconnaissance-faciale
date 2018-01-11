from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import cv2

import tensorflow as tf
import numpy as np

img=[]
vect_img=[]
for i in range(1,201):
    for lettre in ['a','b']:
        img.append(str(i)+lettre)
        
img_path='frontalimages_manuallyaligned_part1/'
#print(img_path+img[2]+'.jpg')


for i in range(0,400):
    image=cv2.imread(img_path+img[i]+'.jpg',0)
    vect_img.append(image)

Label=[]

fichier = open("Label.txt", "r")

while 1:
    txt = fichier.readline()
    if txt =='':
        break
    else:
        Label.append(txt[0])
#print(Label)
fichier.close()

print(np.shape(Label))
print(np.shape(vect_img))
#cv2.imshow('image',vect_img[399])
#cv2.waitKey(0)
#cv2.destroyAllWindows()
