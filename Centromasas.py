import numpy as np
import cv2
from matplotlib import pyplot as plt
from sympy import *
import os
import glob
from datetime import datetime


def line_intersection(line1, line2):

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return x, y



def calcular_distancia (d,m,n,x2,y2):

	x1=0
	y1=0
	y1 = Symbol('y1')
	a,b=solve( ((m*y1+n-x2)**2+(y1-y2)**2)-d**2 ,y1)  
	#print a,b
	y1=int(a[0])
 	x1=m*y1+n

	return (int(y1),int(x1))

def data_process(imgR,imgL):

	stereo = cv2.StereoBM_create(32, 25)
	imgL=cv2.resize(imgL,(400,400))
	imgR=cv2.resize(imgR,(400,400))
	disparity = stereo.compute(imgL,imgR)
	#disparity[disparity>100]=255
	#disparity[disparity<100]=0
	
	return(disparity)

def find_center(stereo):

	(xmax,ymax)=stereo.shape
	count=0
	posx=0
	posy=0
	points=[]
	col_x=[]
	col_y=[]
	for i in range (0,xmax):
	   for n in range (0,ymax):
		if stereo[i,n]>20:                      #Transformacion de la imagen en binaria
						#Cuenta el numero de pixels de colision
						
			stereo[i,n]=255			#Cambia el valor del pixel a blanco puro
		else:
			stereo[i,n]=0			#Cambia el valor del pixel a negro puro
			points.append((i,n))		#Guarda solo los puntos negros
			posx=posx+i
			posy=posy+n
			count+=1
			#posx=posx+(i-200)		#Sumatorio de posicion X
			#posy=posy+(n-200)   		#Sumatorio de posicion Y
			#print (i,n)

	points=np.array(points)				#Transforma lista en array

	if count==0:					#En caso de que no haya pixels de colision evita un div por cero
		count=1
	posx=posx/count				#Calculo de coordenadas del centroide
	posy=posy/count
	#print (posx,posy)
	PCol=count/1600
	#print (PCol)
	return (posx,posy,points)

def regression_lines(points,stereo): 
#Possible optimization not calling stereo.shape[]
	
	if points.size!=320000:									
		[vx,vy,x,y] = cv2.fitLine(points,cv2.DIST_L2,0,0.01,0.01) 	#Realiza esto solo si hay pizels de probabilidad de colision
		#print points.size
		# Now find two extreme points on the line to draw line
		#y=m*x+n
		
       		
		

		lefty = int((-x*vy/vx) + y)					#Busca el extremo izquierdo de la linea
		righty = int(((stereo.shape[1]-x)*vy/vx)+y)			#Busca el extremo derecho de la linea
	
		nx,ny = 1,-vx/vy						#Crea la linea perpendicular
		mag = np.sqrt((1+ny**2))	
		vx2,vy2 = nx/mag,ny/mag

		## Forcing the line to pass the center 
		x=200
		y=200

		lefty2 = int((-x*vy2/vx2) + y)
		righty2 = int(((stereo.shape[1]-x)*vy2/vx2)+y)

		line_params=[vy/vx,((-x*vy/vx) + y),vy2/vx2,((-x*vy2/vx2) + y)] #[m1,n1,m2,n2]

		line1=[(righty,400),(lefty,0)]
		line2=[(righty2,400),(lefty2,0)]
		p= line_intersection(line1, line2)
	else:
		line_params=[0,0,0,0] 
		line1=[(0,400),(400,200)]
		line2=[(400,0),(400,200)]
		p= line_intersection(line1, line2)	
	return line1,line2,p,line_params

def paint_img(line1,line2,image,p_obj):

	img=np.zeros((400,400,1))
	img=cv2.resize(image,(400,400))

	img=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)    
	#cv2.line(img,(line1[0][0],400),(line1[1][0],0),(255,0,0),1)    #RECTA PARALELA
	#cv2.line(img,(line2[0][0],400),(line2[1][0],0),(0,255,0),1)  #RECTA PERPENDICULAR

	cv2.line(img,(0,200),(400,200),(255,255,0),1)   
	cv2.line(img,(200,0),(200,400),(255,255,0),1)   
	
	cv2.circle(img,(p_obj[1],p_obj[0]),10,(0,0,255),2)
	#cv2.circle(img,(int(yz),int(xz)),10,(255,0,255),1)
	return img

def main():
	vectz=10
	p_obj=[0,0]
	general='./DATAc/training'
	Experiments=sorted(glob.glob(general+'/*'))
	number_exp = len(Experiments)
	f=12
	for k in range (0,number_exp):
		general=Experiments[k]
		print general
		frames=sorted(glob.glob(general+'/images/*'))
		number_files = len(frames)
		n=0
		f=f+1
		#if os.path.exists(general+"/LabelsX.txt"):
		#print general +' Already processed'
		#else:
		f1=open(general+"/LabelsX.txt","w")
		f1.write('X Punto objetivo'+'\n')
		f2=open(general+"/LabelsY.txt","w")
		f2.write('Y Punto objetivo'+'\n')
		path='./Dataset/OLD/AlgResult/images_'+str(f)
		try:
	    		#os.mkdir(path)
			#os.mkdir(path)
			for fr in range (0,number_files):
		
				stereo = cv2.imread(frames[n],0)
				stereo=cv2.resize(stereo,(400,400))
		
				posx,posy,regression_points=find_center(stereo)
				p_obj=[posx,posy]
				n=n+1
				cv2.circle(stereo,(p_obj[1],p_obj[0]),10,(255,255,255),2)
				cv2.imshow('img',stereo)
				cv2.waitKey(2)
				#frname='frame_{0:05d}'.format(n)+'.png'
				#cv2.imwrite(path+'/'+frname, img)
				f1.write(str(datetime.now())+'	'+str(p_obj[0])+'\n') #str(p_obj[0])+'  ,'+str(p_obj[1])
				f2.write(str(datetime.now())+'	'+str(p_obj[1])+'\n') #str(p_obj[0])+'  ,'+str(p_obj[1])
			f1.close()
			f2.close()
		except OSError:
	    		print ("Creation of the directory %s failed" % path)

		
if __name__ == "__main__":
    main()







