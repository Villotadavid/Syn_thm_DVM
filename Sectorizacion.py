import numpy as np
import cv2
from matplotlib import pyplot as plt
from sympy import *
import os
import glob
from datetime import datetime
import time


def data_process(imgR,imgL):

	stereo = cv2.StereoBM_create(32, 25)
	imgL=cv2.resize(imgL,(200,200))
	imgR=cv2.resize(imgR,(200,200))
	disparity = stereo.compute(imgL,imgR)
	#disparity[disparity>100]=255
	#disparity[disparity<100]=0
	
	return(disparity)


def pobj(stereo):

	sub_img1 = np.zeros((100,100)) 
        sub_img2 = np.zeros((100,100))
	sub_img3 = np.zeros((100,100))
	sub_img4 = np.zeros((100,100))

        sub_img1 = stereo[0:100,0:100]
	sub_img2 = stereo[100:200,0:100]
	sub_img3 = stereo[0:100,100:200]
	sub_img4 = stereo[100:200,100:200]

	p1,p2,data1,sub_img1=find_center(sub_img1)
	p3,p4,data2,sub_img2=find_center(sub_img2)
	p5,p6,data3,sub_img3=find_center(sub_img3)
	p7,p8,data4,sub_img4=find_center(sub_img4)

	coll1=float(data1.shape[0])/10000
	coll2=float(data2.shape[0])/10000
	coll3=float(data3.shape[0])/10000
	coll4=float(data4.shape[0])/10000

	datalist=[[data1,(p1,p2),coll1],[data2,(p3+100,p4),coll2],[data3,(p5,p6+100),coll3],[data4,(p7+100,p8+100),coll4]]
	datalist.sort(key=lambda x: x[0].shape,reverse = True) #reverse = True,  #key=lambda x: x.shape
	
	
	'''print str(datalist[0][0].shape[0])+'->1'
	print str(datalist[1][0].shape[0])+'->2'
	print str(datalist[2][0].shape[0])+'->3'
	print str(datalist[3][0].shape[0])+'->4'''

	c1=datalist[0][2]
	c2=datalist[1][2]
	c3=datalist[2][2]
	c4=datalist[3][2]

	#print c1,c2,c3,c4

	x1=datalist[0][1][0]
	x2=datalist[1][1][0]
	x3=datalist[2][1][0]

	y1=datalist[0][1][1]
	y2=datalist[1][1][1]
	y3=datalist[2][1][1]

	m1=datalist[0][0].shape[0]
	m2=datalist[1][0].shape[0]
	m3=datalist[2][0].shape[0]

	imgfin = np.zeros((200,200))

	imgfin [0:100,0:100]=sub_img1
	imgfin [100:200,0:100]=sub_img2
	imgfin [0:100,100:200]=sub_img3
	imgfin [100:200,100:200]=sub_img4

	p1,p2=datalist[0][1]
	p3,p4=datalist[1][1]
	p5,p6=datalist[2][1]

	c1=c1*1.5
	#print c1,c2,c3
	#print x1,x2,x3
	print '-------------'
	

	if (x1 and y1 and y2 and x3)==50:
		X=100
		Y=100
	else:
		X=(x1*m1*c1+x2*m2*c2+x3*m3*c3)/(m1+m2+m3)
		Y=(y1*m1*c1+y2*m2*c2+y3*m3*c3)/(m1+m2+m3)

	return(int(X),int(Y),imgfin,p1,p2,p3,p4,p5,p6)


def find_center(stereo):

	(xmax,ymax)=stereo.shape
	#print stereo.shape
	count=0
	countc=0
	posx=0
	posy=0
	points=[]
	col_x=[]
	col_y=[]
	
	#print stereo.max()
	for i in range (0,xmax):
	   for n in range (0,ymax):
		if stereo[i,n]>30:                      #Transformacion de la imagen en binaria
			countc+=1			#Cuenta el numero de pixels de colision		
			stereo[i,n]=255			#Cambia el valor del pixel a blanco puro
			#print i
		else:
			stereo[i,n]=0			#Cambia el valor del pixel a negro puro
			points.append((i,n))		#Guarda solo los puntos negros
			posx=posx+(i+1)			#Sumatorio de posicion X
			posy=posy+(n+1)   		#Sumatorio de posicion Y
			count=count+1	
			#print count		


	points=np.array(points)				#Transforma lista en array

	if count==0:					#En caso de que no haya pixels de colision evita un div por cero
		count=1
	#print (posx)
	posx=posx/count				#Calculo de coordenadas del centroide
	posy=posy/count

	return (posx,posy,points,stereo)





def main():
	vectz=10
	p_obj=[0,0]
	datalist=[]
	general='/home/tev/Desktop/DATAs_X/training'
	Experiments=sorted(glob.glob(general+'/*'))
	number_exp = len(Experiments)
	f=0
	print number_exp
	for k in range (0,number_exp):
		general=Experiments[k]
		print general
		frames=sorted(glob.glob( general+'/images/*'))
		number_files = len(frames)
		n=0
		f=f+1
		po=2
		if po==3: #os.path.exists(general+"/LabelsX.txt"):
			print general +' Already processed'
		else:
			f1=open(general+"/LabelsX.txt","w")
			f1.write('X Punto objetivo'+'\n')
			f2=open(general+"/LabelsY.txt","w")
			f2.write('Y Punto objetivo'+'\n')
			path='./Dataset/OLD/AlgResult/images_'+str(f)
			try:
				#os.mkdir(path)
				for fr in range (0,number_files):
	
					stereo = cv2.imread(frames[n],0)
					stereo = cv2.resize(stereo,(200,200))
		
					Xfinal,Yfinal,imgfin,p1,p2,p3,p4,p5,p6=pobj(stereo)

					img=cv2.circle(imgfin,(Yfinal,Xfinal),10,(225,255,255),2)
					#print Xfinal,Yfinal,p1,p2,p3,p4,p5,p6
					#img=cv2.circle(imgfin,(p1,p2),10,(225,255,255),2)
					#img=cv2.circle(imgfin,(p3,p4),10,(225,255,255),2)
					#img=cv2.circle(imgfin,(p5,p6),10,(225,255,255),2)

					#print 'jola'
					#cv2.imshow('img',img)
					#time.sleep(0.1)
					#cv2.waitKey(1)	
		
					n=n+1

					#frname='frame_{0:05d}'.format(n)+'.png'
					#cv2.imwrite(path+'/'+frname, img)
					SteerX=Xfinal/200.0
					SteerY=Yfinal/200.0
					print SteerX,SteerY
					f1.write(str(SteerX)+'\n') #str(p_obj[0])+'  ,'+str(p_obj[1])
					f2.write(str(SteerY)+'\n') #str(p_obj[0])+'  ,'+str(p_obj[1])
				f1.close()
				f2.close()
			except OSError:
	    			print ("Creation of the directory %s failed" % path)



		
if __name__ == "__main__":
    main()







