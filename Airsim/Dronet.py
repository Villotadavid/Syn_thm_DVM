#!/usr/bin/env python
import rospy

import utils
import numpy as np
import cv2
from datetime import datetime
from sympy import *
import os
import glob
from keras import backend as K
from keras import models
from keras.models import Sequential
from matplotlib import pyplot as plt

TEST_PHASE=0

def pixeles(img,centro,radio=20):
	n=0
	indices=[]
	xmax,ymax=img.shape
	#print xmax
	for x in range (0,ymax):
    		for y in range (0,xmax) :
        		dx = x - centro[0];
        		dy = y - centro[1];
        		distanceSquared = dx * dx + dy * dy;
        		if (distanceSquared <= radio**2):
            			indices.append(img[x,y])
				if img[x,y]==255:
					n=n+1
				

	
	return(indices,n)

prev=np.zeros((200,200,1))
K.set_learning_phase(TEST_PHASE)

        # Load json and create model
model1 = utils.jsonToModel('./model_struct_X.json')
model2 = utils.jsonToModel('./model_struct_Y.json')
       # Load weights
model1.load_weights('./weights_X.h5')
model2.load_weights('./weights_Y.h5')
print("Loaded model from {}".format('./weights_X.h5'))

model1.compile(loss='mse', optimizer='sgd')
model2.compile(loss='mse', optimizer='sgd')

print 'Process start'
general='/home/tev/Desktop/Accuracy/Exp69' 
f1=open("Log.txt","w")
f1.write('Punto objetivo'+'\n')
f3=open(general+"/AccuracyF.txt","w")
f3.write('collision_points'+'\n')       

#general='/home/tev/Desktop/DATAr_X/repost/training/Exp10'

print general
frames=sorted(glob.glob( general+'/images/*'))
number_files = len(frames)
print number_files

layer_outputs = [layer.output for layer in model1.layers[:25]] # Extracts the outputs of the top 12 layers
activation_model1 = models.Model(inputs=model1.input, outputs=layer_outputs[5]) 
activation_model2 = models.Model(inputs=model1.input, outputs=layer_outputs[13])
activation_model3 = models.Model(inputs=model1.input, outputs=layer_outputs[24])
r=35
for fr in range (0,number_files):

	img=stereo= cv2.imread(frames[fr],0)
	img = utils.callback_img(img, (320,240), (200,200),0,0)
	stereo=cv2.resize(stereo,(200,200))
	#img=img+0.1

	'''activations1 = activation_model1.predict(img[None])
	first_layer_activation1= activations1[0]

	activations2 = activation_model2.predict(img[None])
	first_layer_activation2= activations2[0]

	activations3 = activation_model3.predict(img[None])
	first_layer_activation3= activations3[0]'''
	

	outs1 = model1.predict_on_batch(img[None])
	outs2 = model2.predict_on_batch(img[None])
	   
	steer, coll = outs1[0][0], outs2[0][0]
	#print ACT
	#print steer, coll
	x=int(steer*200)
	y=int(coll*200)
	
	#f1=open("/home/tev/Desktop/Log.txt","w")
	#f1.write(str(datetime.now())+'	'+str(steer)+'	'+str(coll)+'\n')
	p_obj=[x,y] 
  
	#n=n+1
	#img=img*4
	t, stereo = cv2.threshold(stereo, 15, 255, cv2.THRESH_BINARY)
	stereo = cv2.GaussianBlur(stereo, (7, 7), 3)
	t, stereo = cv2.threshold(stereo, 0, 255, cv2.THRESH_BINARY)
	
	p,num=pixeles(stereo,p_obj,35)
	collision=num/(len(p)*1.0)
	cv2.circle(stereo,(x,y),5,(255,0,255),4)
	cv2.circle(stereo,(x,y),35,(255,0,255),2)
	cv2.imshow('img',stereo)
	cv2.waitKey(2)  
    

	#print first_layer_activation2.shape
	#plt.subplot(2, 3, 1).set_title('Profundidad')
	#plt.imshow(img[:,:,0], cmap='pink')
	'''plt.subplot(2, 2, 1).set_title('Prediction')
	plt.imshow(img[:,:,0], cmap='pink')
	plt.subplot(2, 2, 2).set_title('Normalizacion 1 ')
	plt.imshow(first_layer_activation1[:,:,8],cmap='binary')
	plt.subplot(2, 2, 3).set_title('Normalizacion 4 ')
	plt.imshow(first_layer_activation1[:,:,30],cmap='gray')
	plt.subplot(2, 2, 4).set_title('Normalizacion 6 ')
	plt.imshow(first_layer_activation3[:,:,125],cmap='binary')'''


	#plt.savefig('frame_{0:05d}'.format(fr)+'.png')

	#print x,y,collision
	f3.write(str(collision)+'\n')
	f1.write(str(x)+' '+str(y)+'\n')

plt.show()
f1.close()
f3.close()
