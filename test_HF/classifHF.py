from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import cv2

import tensorflow as tf
import numpy as np
import time

"""
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
"""
def neural_network_model(data): 
    # input_data * weights + biases 
    hidden_l1 = {'weights': tf.Variable(tf.random_normal([93600, n_nodes_hl1])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))} 
    hidden_l2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))} 
    hidden_l3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    output_l = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))} 

    l1 = tf.add(tf.matmul(data, hidden_l1['weights']), hidden_l1['biases']) 
    l1 = tf.nn.relu(l1) 

    l2 = tf.add(tf.matmul(l1, hidden_l2['weights']), hidden_l2['biases']) 
    l2 = tf.nn.relu(l2) 

    l3 = tf.add(tf.matmul(l2, hidden_l3['weights']), hidden_l3['biases']) 
    l3 = tf.nn.relu(l3) 

    output = tf.add(tf.matmul(l3, output_l['weights']), output_l['biases']) 
    return output 

def train_neural_network(x): 
    prediction = neural_network_model(x)
   
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))#a verif # v1.0 changes 

    # optimizer value = 0.001, Adam similar to SGD 
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    epochs_no = 2
    
    with tf.Session() as sess: 
        sess.run(tf.global_variables_initializer()) # v1.0 changes 
        # training 
        for epoch in range(epochs_no): 
            epoch_loss = 0 
            for p in range(8): 
                epoch_x=epoch_img[:,:,p]
                epoch_y=epoch_label[:,:,p]
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y}) # code that optimizes the weights & biases 
                epoch_loss += c
                #_, c = sess.run([optimizer, cost], feed_dict = {x: vect_img, y: Label}) # code that optimizes the weights & biases 

            print('Epoch', epoch, 'completed out of', epochs_no, 'loss:', epoch_loss) 
        # testing 
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        valeur = tf.argmax(prediction, 1).eval({x:vect_img , y: Label})
        print('valeur',valeur)
        #print('valeur',tf.argmax(prediction, 1).eval({x:vect_img , y: Label}))
        estim = tf.argmax(y, 1).eval({x:vect_img , y: Label});
        print('estim', estim)
        #print('estim',tf.argmax(y, 1).eval({x:vect_img , y: Label}))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float')) 
        print('Accuracy:', accuracy.eval({x:vect_img , y: Label}))
        #print(correct)
        print('C EST LE TEST DE LA VERITE')
        ind_false = np.nonzero((valeur-estim))
        print(ind_false)


train_neural_network(x)















