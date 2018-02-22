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


fold=5

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite('test.jpg',frame)
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

image=cv2.imread('test.jpg',0)



'''
vect_img=np.load('Img_flatten.npy');
Label=np.load('Label_test.npy');

print(np.shape(vect_img))
print(np.shape(vect_img[0:1,:]))
print((vect_img[0,:]))
print((vect_img[0:1,:]))
'''

print(np.shape(image))
image=cv2.resize(image, (360,260))
vect_img=np.reshape(image,[1,360*260])

n_nodes_hl1 = 50 


n_classes = 2 
nb_pixel=360*260

# input feature size = 360*260 pixels = 93600 
x = tf.placeholder('float', [None, nb_pixel]) 
y = tf.placeholder('float')


def neural_network_model(data):
    # input_data * weights + biases 
    hidden_l1 = {'weights': tf.Variable(tf.random_normal([93600, n_nodes_hl1])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))} 
   
        
    output_l = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))} 

    
    l1 = tf.add(tf.matmul(data, hidden_l1['weights']), hidden_l1['biases']) 
    l1 = tf.nn.relu(l1)

        

    output = tf.add(tf.matmul(l1, output_l['weights']), output_l['biases'])


    return output 

cross_val=0

def train_neural_network(x):
    prediction = neural_network_model(x)
       
    #summary_writer=tf.summary.FileWriter(logdir,tf.get_default_graph())

        
    epochs_no = 10
    
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer()) # v1.0 changes
        saver.restore(sess, "./my_10fold_model/")
        # testing 7
        #print('valeur',tf.argmax(prediction, 1).eval({x:vect_img}))
        
        print('valeur',tf.argmax(prediction, 1).eval({x:vect_img}))
        #print('true',Label)
        #print('estim',tf.argmax(y, 1).eval({y: Label}))
           
            
        #summary_writer.close()
    return()
    


'''
ACC=np.zeros(5)
for i in range(5):
    ACC[i]=train_neural_network(x)
    print('iteration : ',i)

for i in range(5):
    print(ACC[i])

print('accuracy = ',np.mean(ACC))
'''

train_neural_network(x)
#print(" Mean accuracy : ",np.mean(ACC))


elapsed = time.time() - t
print("elapsed time",elapsed)


