from cv_bridge import CvBridge, CvBridgeError
from keras.models import model_from_json
import cv2
import numpy as np
import rospy

#prev=np.zeros((200,200,1))

def flowToDisplay (flow):

	rgb=np.zeros((200,200,3),dtype=np.float32)
	
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	hsv = rgb
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,1] = 1
	hsv[...,2] = cv2.normalize(mag,None,0,1,cv2.NORM_MINMAX) 
	bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

	return bgr


bridge = CvBridge()

def callback_img(data, target_size, crop_size, rootpath, save_img,prev):
    try:
        image_type = data.encoding
        img = bridge.imgmsg_to_cv2(data, image_type)
	#prev= bridge.imgmsg_to_cv2(data, image_type)
    except CvBridgeError, e:
        print e
    
    
    #prev=prev*255
    #cv2.imshow('image',img)
    #cv2.waitKey(1)
    img = cv2.resize(img, target_size)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = central_image_crop(img, crop_size[0], crop_size[1])
    #h,w= img.shape[:2]
    
    #print (h,w)
    #uflow=cv2.calcOpticalFlowFarneback(prev,img,None,0.4,1,12,2,8,1,0)#
    #uflow=np.array(uflow)#
    #rgb=flowToDisplay(uflow)#
    #img=img/255
    #Definir dimensiones
   
    #rgb=rgb*255
    #print rgb
    #image_4Ch=np.zeros((h,w,4))
    #image_4Ch=np.concatenate((img,rgb),axis=2)
    #cv2.imshow('image',image_4Ch)
    #cv2.waitKey(1)

    


    if rootpath and save_img:
        temp = rospy.Time.now()
        cv2.imwrite("{}/{}.jpg".format(rootpath, temp), rgb) #Changed to save OF

    return np.asarray( img, dtype=np.float32)* np.float32(1.0/255.0)


def central_image_crop(img, crop_width, crop_heigth):
    """
    Crops the input PILLOW image centered in width and starting from the bottom
    in height.
    Arguments:
        crop_width: Width of the crop
        crop_heigth: Height of the crop
    Returns:
        Cropped image
    """
    half_the_width = img.shape[1] / 2
    img = img[(img.shape[0] - crop_heigth): img.shape[0],
              (half_the_width - (crop_width / 2)): (half_the_width + (crop_width / 2))]
    img = img.reshape(img.shape[0], img.shape[1], 1)
    return img

def jsonToModel(json_model_path):
    with open(json_model_path, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)

    return model
