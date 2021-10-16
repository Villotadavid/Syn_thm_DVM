import threading
import time
import paramiko
import cv2
import numpy as np
from multiprocessing import Process
from datetime import datetime
from matplotlib import pyplot as plt
import os

def ssh_client1():
    ip='10.0.11.2'
    print ("Connecting to: 10.0.11.2")
    ssh_client1=paramiko.SSHClient()
    ssh_client1.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client1.connect(hostname=ip,username='pi',password='raspberry')
    n=0
    ftp_client1=ssh_client1.open_sftp()
    stdin,stdout,stderr=ssh_client1.exec_command('echo Connection_OK')
    print (stdout.read())
    return (ssh_client1,ftp_client1)
  

def ssh_client2():
    ip='10.0.12.2'
    print ("Connecting to: 10.0.12.2")
    ssh_client2=paramiko.SSHClient()
    ssh_client2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client2.connect(hostname=ip,username='pi',password='raspberry')
    n=0
    stdin,stdout,stderr=ssh_client2.exec_command('echo Connection_OK')
    print (stdout.read())
    ftp_client2=ssh_client2.open_sftp()
    return (ssh_client2,ftp_client2)
 
 
def photo1(ssh_client,ftp_client,n):

    stdin,stdout,stderr=ssh_client.exec_command('raspistill -n -o foo.jpg --exposure sports --timeout 2 -w 1200 -h 1200 -q 75 -rot 270')
    time.sleep(20)
    ftp_client.get('foo.jpg','foo'+n+'.jpg')

    return(0)

def photo2(ssh_client,ftp_client,n):

    stdin,stdout,stderr=ssh_client.exec_command('raspistill -n -o foo.jpg --exposure sports --timeout 2 -w 1200 -h 1200 -q 75 -rot 90')
    time.sleep(20)
    ftp_client.get('foo.jpg','foo'+n+'.jpg')

    return(0)

def Close(ftp_client):             
    ftp_client.close()
    print('Connection finished')
    return(0)



ssh_client1,ftp_client1=ssh_client1()
ssh_client2,ftp_client2=ssh_client2()

n=0
print (str(datetime.now()))
for n in range (0,5):
    x1 = threading.Thread(target=photo1, args=(ssh_client1,ftp_client1,'1'+str(n)))
    x2 = threading.Thread(target=photo2, args=(ssh_client2,ftp_client2,'2'+str(n)))
    x1.start()
    x2.start()
    x1.join()
    x2.join()
    print (str(datetime.now()))
    '''photo1(ssh_client1,ftp_client1,'1'+str(1))
time.sleep(10)
photo2(ssh_client2,ftp_client2,'2'+str(1))'''

Close(ftp_client1)
Close(ftp_client2)

print ('Starting stereo process')

imgL = cv2.imread('foo22.jpg',0)
imgR = cv2.imread('foo11.jpg',0)

stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()

#os.remove("foo21.jpg")
#os.remove("foo11.jpg")

'''
    x1 = threading.Thread(target=photo, args=(ssh_client1,ftp_client1,'1'+str(n)))
    x2 = threading.Thread(target=photo, args=(ssh_client2,ftp_client2,'2'+str(n)))
    x1.start()
    x2.start()
    x1.join()
    x2.join()
img=cv2.imread('foo'+str(n)+'.jpg',0)
img=cv2.circle(img,(800,800),100,(0,0,0),8)
time=strftime("%Y-%m-%d %H:%M:%S",gmtime())
img=cv2.putText(img,time,(200,200),cv2.FONT_HERSHEY_SIMPLEX,4,8)
cv2.imwrite('fooR'+str(n)+'.jpg',img)
#photo(ssh_client2,ftp_client2,'2'+str(1))
for n in range (0,10):
    

    photo(ssh_client1,ftp_client1,'1'+str(n))
    time.sleep(20)
    photo(ssh_client2,ftp_client2,'2'+str(n))
    time.sleep(20)
print (str(datetime.now()))'''


