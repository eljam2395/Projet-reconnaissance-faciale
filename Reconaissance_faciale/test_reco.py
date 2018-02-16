from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import cv2

import tensorflow as tf
import numpy as np
import time
from datetime import datetime

t = time.time()

now=datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir="tf_logs"
logdir="{}/run-{}/".format(root_logdir,now)


vect_img=np.load("Img_flatten_RF.npy")

nb_pixel=240*320

fold=5

epoch_img=np.ones((140,nb_pixel,fold),float)
print(np.shape(epoch_img))

print(np.shape(vect_img[0:140,:]))

epoch_img[:,:,0]=vect_img[0:140,:]
epoch_img[:,:,1]=vect_img[140:280,:]
epoch_img[:,:,2]=vect_img[280:420,:]
epoch_img[:,:,3]=vect_img[420:560,:]
epoch_img[:,:,4]=vect_img[560:700,:]



Label=np.load('Label_RF.npy');


epoch_label=np.ones((140,50,fold),float)
print(np.shape(Label[0:140]))
print(np.shape(epoch_label))


epoch_label[:,:,0]=(Label[0:140])
epoch_label[:,:,1]=(Label[140:280])
epoch_label[:,:,2]=(Label[280:420])
epoch_label[:,:,3]=(Label[420:560])
epoch_label[:,:,4]=(Label[560:700])


print(np.shape(Label))

#print(np.shape(tab_img[1][:]))


n_nodes_hl1 = 500 
n_nodes_hl2 = 500 
n_nodes_hl3 = 500

n_classes = 50 


# input feature size = 360*260 pixels = 93600 
x = tf.placeholder('float', [None, nb_pixel]) 
y = tf.placeholder('float') 

def neural_network_model(data):
        # input_data * weights + biases 
    hidden_l1 = {'weights': tf.Variable(tf.random_normal([nb_pixel, n_nodes_hl1])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))} 
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
       
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))#a verif # v1.0 changes 
    #mse_summary=tf.summary.scalar("mse",cost)
    #summary_writer=tf.summary.FileWriter(logdir,tf.get_default_graph())

        # optimizer value = 0.001, Adam similar to SGD 
    optimizer = tf.train.AdamOptimizer().minimize(cost)
        
    epochs_no = 10
    #config = tf.ConfigProto()
    #config.gpu_options.allocator_type = 'BFC'
    
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        #writer = tf.summary.FileWriter("output", sess.graph)
        #summaries = tf.summary.merge_all()

        sess.run(tf.global_variables_initializer()) # v1.0 changes
        for p in range(601):
            _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_img[:,:,p%5], y: epoch_label[:,:,p%5]}) # code that optimizes the weights & biases 
            epoch_loss = c
            if(p%100==0):
                print("iter :",p," loss :",epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        ACC=np.zeros(5)
        for cross_val in range(5):
            #print('valeur',tf.argmax(prediction, 1).eval({x:vect_img , y: Label}))
            #print('estim',tf.argmax(y, 1).eval({x:vect_img , y: Label}))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            ACC[cross_val]=accuracy.eval({x:vect_img , y: Label})
            
        return(ACC)
    


'''
ACC=np.zeros(5)
for i in range(5):
    ACC[i]=train_neural_network(x)
    print('iteration : ',i)

for i in range(5):
    print(ACC[i])

print('accuracy = ',np.mean(ACC))
'''
Acc=train_neural_network(x)
print("precision = ",np.mean(Acc))
'''
ACC=train_neural_network(x)
print(" Mean accuracy : ",np.mean(ACC))

'''
elapsed = time.time() - t
print("elapsed time",elapsed)






