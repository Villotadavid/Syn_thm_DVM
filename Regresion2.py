import numpy as np
import cv2
from matplotlib import pyplot as plt
from sympy import *
import os
import glob
from scipy.optimize import fsolve
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

	try:
		a,b=solve( ((200-(m*y1+n))**2+(200-(y1))**2)-d**2 ,y1)

		if isinstance(a, complex)==False:
			y1=int(a[0])
	 		x1=int(y1*m+n)
		else:
			y1=200
			x1=200

	except:
		if m>=1000:
			if x2<=200:

				y1=200
				x1=x2+d
			else :

				y1=200
				x1=x2-d
		elif m==0:
			if x2<=200:

				x1=200
				y1=x2-d
			else :

				x1=200
				y1=x2+d
		else:
			y1=200
			x1=200
	
	return (int(y1),int(x1))

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



	c1=datalist[0][2]
	c2=datalist[1][2]
	c3=datalist[2][2]

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

	Gain=1.2

	if (x1 and y1 and y2 and x3)==50:
		X=100
		Y=100
	else:
		X=(x1*m1*c1+x2*m2*c2+x3*m3*c3)/(m1+m2+m3)
		Y=(y1*m1*c1+y2*m2*c2+y3*m3*c3)/(m1+m2+m3)

	return(int(X),int(Y),imgfin)


def find_center(stereo):

	(xmax,ymax)=stereo.shape
	count=0
	countc=0
	posx=0
	posy=0
	points=[]
	col_x=[]
	col_y=[]
	
	for i in range (0,xmax):
	   for n in range (0,ymax):
		if stereo[i,n]>5:                      #Transformacion de la imagen en binaria
			#countc+=1			#Cuenta el numero de pixels de colision		
			stereo[i,n]=255			#Cambia el valor del pixel a blanco puro
			points.append((i,n))		#Guarda solo los puntos blancos
		else:
			stereo[i,n]=0			#Cambia el valor del pixel a negro puro
			
			posx=posx+(i)			#Sumatorio de posicion X
			posy=posy+(n)   		#Sumatorio de posicion Y
			count=count+1			


	points=np.array(points)				#Transforma lista en array
	#print count
	if count==0:					#En caso de que no haya pixels de colision evita un div por cero
		count=1
	posx=posx/count				#Calculo de coordenadas del centroide
	posy=posy/count

	return (posx,posy,points,stereo)

def regression_lines(points,stereo): 
#Possible optimization not calling stereo.shape[]

	if points.size!=0:	
										
		[vx,vy,x,y] = cv2.fitLine(points,cv2.DIST_L2,0,0.01,0.01)

		lefty = int((-x*vy/vx) + y)					
		righty = int(((stereo.shape[1]-x)*vy/vx)+y)			
		line1=[(righty,400),(lefty,0)]				
		nx,ny=1,-vx/vy
		mag = np.sqrt((1+ny**2))
		vx2,vy2 = nx/mag,ny/mag
		x2=200
		y2=200

		if vx2==0:
			
			a=9999999
			lefty2 = int((-x2*a) + y2)					
			righty2 = int(((stereo.shape[1]-x2)*a)+y2)
			line2=[(righty2,400),(lefty2,0)]

			p= line_intersection(line1, line2)
			line_params=[0,0,0,0]
 			line_params=[vy/vx,((-x*vy/vx) + y),a,((-x2*a) + y2)]

		elif vx2==1:

			a=0
			lefty2 = int((-x2*a) + y2)					
			righty2 = int(((stereo.shape[1]-x2)*a)+y2)
			line2=[(righty2,400),(lefty2,0)]

			p= line_intersection(line1, line2)
			line_params=[0,0,0,0]
			line_params=[vy/vx,((-x*vy/vx) + y),a,((-x2*a) + y2)]

		else:
			
			lefty2 = int((-x2*vy2/vx2) + y2)					
			righty2 = int(((stereo.shape[1]-x2)*vy2/vx2)+y2)
			line2=[(righty2,400),(lefty2,0)]

			p= line_intersection(line1, line2)
			line_params=[0,0,0,0]
			line_params=[vy/vx,((-x*vy/vx) + y),vy2/vx2,((-x2*vy2/vx2) + y2)]
	else:

		line_params=[0,0,0,0] 
		line1=[(0,200),(400,200)]
		line2=[(400,0),(400,200)]
		p= line_intersection(line1, line2)
	

	if line1[0][0]>=10000:
		line1[0]=(10000,line1[0][1])
	elif line2[0][0]>=10000:
		line2[0]=(10000,line2[0][1])	
	elif line1[0][0]<=-10000:
		line1[0]=(-10000,line1[0][1])
	elif line2[0][0]<=-10000:
		line2[0]=(-10000,line2[0][1])


	if line1[1][0]>=10000:
		line1[1]=(10000,line1[1][1])
	elif line2[1][0]>=10000:
		line2[1]=(10000,line2[1][1])	
	elif line1[1][0]<=-10000:
		line1[1]=(-10000,line1[1][1])
	elif line2[1][0]<=-10000:
		line2[0]=(-10000,line2[0][1])

	return line1,line2,p,line_params

def paint_img(line1,line2,image,p_obj):

	img=np.zeros((400,400,1))
	img=cv2.resize(image,(400,400))

	img=cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  
	cv2.line(img,(line1[0]),(line1[1]),(255,0,0),1)    #RECTA PARALELA
	cv2.line(img,(line2[0]),(line2[1]),(0,255,0),1)  #RECTA PERPENDICULAR

	cv2.line(img,(0,200),(400,200),(255,255,0),1)   
	cv2.line(img,(200,0),(200,400),(255,255,0),1)   
	
	cv2.circle(img,(p_obj[0],p_obj[1]),10,(0,0,255),2)

	return img

def main():
	vectz=10
	p_obj=[0,0]
	datalist=[]
	general='./DATAr/validation'
	Experiments=sorted(glob.glob(general+'/*'))
	number_exp = len(Experiments)
	f=0
	d=20
	for k in range (0,number_exp):
		general=Experiments[k]
		print general
		frames=sorted(glob.glob( general+'/images/*'))
		number_files = len(frames)
		n=0
		f=f+1
		po=2
		if po==3:
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
					stereo=cv2.resize(stereo,(400,400))
					posx,posy,points,stereo=find_center(stereo)
					#print posx,posy
					line1,line2,p,line_params=regression_lines(points,stereo)
					w=200
					img=stereo
				
					if len(points)!=0 or line1!=[(0,200),(400,200)]:
					
						d=180*(len(points))/100000
						y1,x1=calcular_distancia (d,line_params[2],line_params[3],p[0],p[1])
						
					else:
						x1=200
						y1=200

					p_obj=[x1,y1]
					Xfinal=x1
					Yfinal=y1

					if Xfinal>=400:
						Xfinal>=400
					elif Xfinal<=0:
						Xfinal>=0

					if Yfinal>=400:
						Yfinal>=400
					elif Yfinal<=0:
						Yfinal>=0
					#p_obj=[200,200]
					#img=paint_img(line1,line2,stereo,p_obj)

					#img=cv2.circle(img,(posy,posx),10,(255,255,255),2)

					n=n+1
					#cv2.imshow('img',img)
					#cv2.waitKey(2)
					#frname='frame_{0:05d}'.format(n)+'.png'
					#path=frames[n]+'/images/'
					#cv2.imwrite(path+frname, img)
					SteerX=p_obj[0]/400.0
					SteerY=p_obj[1]/400.0					
					#SteerX=(p_obj[0]-200)/200.0
					#SteerY=(p_obj[1]-200)/200.0
					f1.write(str(SteerX)+'\n') #str(p_obj[0])+'  ,'+str(p_obj[1])
					f2.write(str(SteerY)+'\n') #str(p_obj[0])+'  ,'+str(p_obj[1])
				f1.close()
				f2.close()
			except OSError:
	    			print ("Creation of the directory %s failed" % path)



		
if __name__ == "__main__":
    main()







