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
from sklearn.utils import shuffle

t = time.time()

now=datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir="tf_logs"
logdir="{}/run-{}/".format(root_logdir,now)

vect_img=np.load('Img_flatten.npy');
Label=np.load('Label_test.npy');

vect_img, Label = shuffle(vect_img, Label, random_state=0)
fold=10

epoch_img=np.ones((40,93600,fold),float)
print(np.shape(epoch_img))


epoch_img[:,:,0]=(vect_img[0:40,:])
epoch_img[:,:,1]=(vect_img[40:80,:])
epoch_img[:,:,2]=(vect_img[80:120,:])
epoch_img[:,:,3]=(vect_img[120:160:])
epoch_img[:,:,4]=(vect_img[160:200,:])
epoch_img[:,:,5]=(vect_img[200:240,:])
epoch_img[:,:,6]=(vect_img[240:280,:])
epoch_img[:,:,7]=(vect_img[280:320,:])
epoch_img[:,:,8]=(vect_img[320:360,:])
epoch_img[:,:,9]=(vect_img[360:400,:])



epoch_label=np.ones((40,2,fold),float)
print(np.shape(epoch_label))


epoch_label[:,:,0]=(Label[0:40,:])
epoch_label[:,:,1]=(Label[40:80,:])
epoch_label[:,:,2]=(Label[80:120,:])
epoch_label[:,:,3]=(Label[120:160:])
epoch_label[:,:,4]=(Label[160:200,:])
epoch_label[:,:,5]=(Label[200:240,:])
epoch_label[:,:,6]=(Label[240:280,:])
epoch_label[:,:,7]=(Label[280:320,:])
epoch_label[:,:,8]=(Label[320:360,:])
epoch_label[:,:,9]=(Label[360:400,:])
print(np.shape(Label))
print(np.shape(vect_img))
#print(np.shape(tab_img[1][:]))


n_nodes_hl1 = 500 
n_nodes_hl2 = 500 
n_nodes_hl3 = 500

n_classes = 2 


# input feature size = 360*260 pixels = 93600 
x = tf.placeholder('float', [None, 93600]) 
y = tf.placeholder('float') 

def neural_network_model(data): 
    # input_data * weights + biases 
    hidden_l1 = {'weights': tf.Variable(tf.random_normal([93600, n_nodes_hl1])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))} 
    hidden_l2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))} 
    hidden_l3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    
    output_l = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))} 

    l1 = tf.add(tf.matmul(data, hidden_l1['weights']), hidden_l1['biases'])
    #l1=tf.sigmoid(l1)
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, hidden_l2['weights']), hidden_l2['biases']) 
    #l2=tf.sigmoid(l2)
    l2 = tf.nn.relu(l2) 

    l3 = tf.add(tf.matmul(l2, hidden_l3['weights']), hidden_l3['biases']) 
    #l3=tf.sigmoid(l3)
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_l['weights']), output_l['biases'])


    return output 

def train_neural_network(x): 
    prediction = neural_network_model(x)
   
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))#a verif # v1.0 changes 
    mse_summary=tf.summary.scalar("mse",cost)
    summary_writer=tf.summary.FileWriter(logdir,tf.get_default_graph())
    # optimizer value = 0.001, Adam similar to SGD 
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    epochs_no = 10
    
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ACC=np.zeros(fold)
        sess.run(tf.global_variables_initializer()) # v1.0 changes
        step=1
        for cross_val in range(fold): 
            # training
            print('iter : ',cross_val)
            array=np.linspace(0,fold-1,fold)
            np.delete(array,cross_val)
            for epoch in array:
                epoch=int(epoch)
                epoch_x=epoch_img[:,:,epoch]
                epoch_y=epoch_label[:,:,epoch]
                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y}) # code that optimizes the weights & biases 
                epoch_loss = c
                #_, c = sess.run([optimizer, cost], feed_dict = {x: vect_img, y: Label}) # code that optimizes the weights & biases 
                print('Epoch', epoch, 'completed out of', epochs_no, 'loss:', epoch_loss) 
            # testing 7
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            print('valeur',tf.argmax(prediction, 1).eval({x:epoch_img[:,:,cross_val] , y: epoch_label[:,:,cross_val]}))
            print('estim',tf.argmax(y, 1).eval({x:epoch_img[:,:,cross_val] , y: epoch_label[:,:,cross_val]}))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            ACC[cross_val]=accuracy.eval({x:epoch_img[:,:,cross_val] , y: epoch_label[:,:,cross_val]})
            print('Accuracy:',ACC[cross_val])
            summary_writer.add_summary(mse_summary.eval(feed_dict={x:epoch_img[:,:,cross_val],y:epoch_label[:,:,cross_val]}),step)
            
            step=step+1
        summary_writer.close()
        saver.save(sess, './my_10fold_model/')
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

ACC=train_neural_network(x)
print(" Mean accuracy : ",np.mean(ACC))

elapsed = time.time() - t
print("elapsed time",elapsed)








