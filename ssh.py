import time
import paramiko
import cv2
import numpy as np


def ssh_client():
    ip='rpimain'
    print ("Connecting to: RPimain")
    ssh_client=paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=ip,username='pi',password='raspberry')
    n=0
    ftp_client=ssh_client.open_sftp()
    stdin,stdout,stderr=ssh_client.exec_command('echo Connection_OK')
    print (stdout.read())
    #stdin,stdout,stderr=ssh_client.exec_command('cd Desktop')
    #stdin,stdout,stderr=ssh_client.exec_command('ls -a')
    #print (stdout.read())
    return (ssh_client,ftp_client)

def photo(ssh_client,ftp_client):
    ftp_client.get('Desktop/foo21.jpg','foo21.jpg')
    ftp_client.get('Desktop/foo11.jpg','foo11.jpg')
    return(0)

def Close(ftp_client):             
    ftp_client.close()
    print('Connection finished')
    return(0)


ssh_client,ftp_client=ssh_client()
photo(ssh_client,ftp_client)
Close(ftp_client)

'''#paramiko.util.log_to_file("filename.log")
ip='192.168.1.64'
ssh_client=paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh_client.connect(hostname='rpimain',username='pi', password='raspberry')
print 'hoelo'
stdin,stdout,stderr=ssh_client.exec_command('echo hello')
print (stdout.read())'''


