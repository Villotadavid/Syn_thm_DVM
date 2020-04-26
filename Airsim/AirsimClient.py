import setup_path 
import airsim
import time
import os
import numpy as np
import cv2
import glob
#from tensorflow import keras
#from keras import backend as K
#from keras import models
#from keras.models import Sequential
from matplotlib import pyplot as plt
import utils
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# connect to the AirSim simulator 
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = airsim.CarControls()

        # Load json and create model
#model1 = utils.jsonToModel('./model_struct_X.json')
#model2 = utils.jsonToModel('./model_struct_Y.json')
       # Load weights
#model1.load_weights('./weights_X.h5')
#model2.load_weights('./weights_Y.h5')
#print("Loaded model from {}".format('./weights_X.h5'))

#model1.compile(loss='mse', optimizer='sgd')
#model2.compile(loss='mse', optimizer='sgd')




for idx in range(2):
    # get state of the car
    

    filename='C:/Users/usuario/Documents/GitHub/TFM/image'+str(idx)
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)]) 
    response=responses[0]
    
    img = airsim.list_to_2d_float_array(response.image_data_float, response.width, response.height)

    
    img *= 255.0/img.max()
    img = (255-img)
    print (np.amax(img),np.mean(img))
    cv2.calcHist([img],[0],None,[65600],[0,65600])
    plt.hist(img.ravel(),65600,[0,65600])
    plt.show()
    cv2.imshow('img',img)
    cv2.waitKey(10)
    
    a, img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)

 
   
    coll=cv2.countNonZero(img)
    img = utils.callback_img(img, (320,240), (200,200),0,0)
    outs1 = model1.predict_on_batch(img[None])
    outs2 = model2.predict_on_batch(img[None])
    x=outs1
    y=outs2
    print (x,y)
    car_controls.throttle = coll/40000
    car_controls.steering = 0
    client.setCarControls(car_controls)


car_controls.brake = 1
client.setCarControls(car_controls)
print("Apply brakes")
time.sleep(3)   # let car drive a bit       


#restore to original state
client.reset()

client.enableApiControl(False)


            

