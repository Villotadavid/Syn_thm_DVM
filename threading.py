import threading
import time
import paramiko
import time
from time import strftime,gmtime
import cv2
import numpy as np
import cv2
#from matplotlib import pyplot as plt

def ssh_client1():
    ip='10.0.11.2'
    print ("Connecting to: 10.0.11.2")
    ssh_client=paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=ip,username='pi',password='raspberry')
    n=0
    ftp_client=ssh_client.open_sftp()
    stdin,stdout,stderr=ssh_client.exec_command('echo Connection_OK')
    print (stdout.read())
    
    for n in range(0,10):
        print('Foto de 1')
        stdin,stdout,stderr=ssh_client.exec_command('raspistill -n -o foo.jpg --exposure sports --timeout 1')
        ftp_client.get('foo.jpg','foo'+str(n)+'.jpg')
        img=cv2.imread('foo'+str(n)+'.jpg',0)
        img=cv2.circle(img,(800,800),100,(0,0,0),8)
        time=strftime("%Y-%m-%d %H:%M:%S",gmtime())
        img=cv2.putText(img,time,(200,200),cv2.FONT_HERSHEY_SIMPLEX,4,8)
        cv2.imwrite('fooR'+str(n)+'.jpg',img)
        n=n+1
        
    ftp_client.close()
    print('Connection finished')
    return(0)


def ssh_client2():
    ip='10.0.12.2'
    print ("Connecting to: 10.0.12.2")
    ssh_client=paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=ip,username='pi',password='raspberry')
    n=0
    ftp_client=ssh_client.open_sftp()
    stdin,stdout,stderr=ssh_client.exec_command('echo Connection_OK')
    print (stdout.read())
    
    for n in range(0,10):
        print('Foto de 2')
        stdin,stdout,stderr=ssh_client.exec_command('raspistill -n -o foo.jpg --exposure sports --timeout 1')
        print('Foto de 2')
        ftp_client.get('foo.jpg','foo'+str(n)+'.jpg')
        img=cv2.imread('foo'+str(n)+'.jpg',0)
        img=cv2.circle(img,(800,800),100,(0,0,0),8)
        time=strftime("%Y-%m-%d %H:%M:%S",gmtime())
        img=cv2.putText(img,time,(200,200),cv2.FONT_HERSHEY_SIMPLEX,4,8)
        cv2.imwrite('fooL'+str(n)+'.jpg',img)
        n=n+1
        
    ftp_client.close()
    print('Connection finished')
    return(0)

#ssh_client('10.0.12.2')
#ssh_client('10.0.11.2')

#t = threading.Thread(name='ssh_client', target=ssh_client('10.0.12.2'))
#w = threading.Thread(name='ssh_client1', target=ssh_client('10.0.11.2'))
#w2 = threading.Thread(target=ssh_client) # use default name

threading.Thread(target = ssh_client2).start()
threading.Thread(target = ssh_client1).start()


