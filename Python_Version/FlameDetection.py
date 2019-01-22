# import the necessary packages
import numpy as np
import cv2
import math

import sys
sys.path.append('./../CONSTANTS')
import CONSTANTS as CON

def Motion_Detaction(prev , next):
	
	flow = cv2.calcOpticalFlowFarneback(prev,next,None, 0.5,3,15,3,5,1.2,0)
	
	mag,ang = cv2.cartToPolar(flow[...,0],flow[...,1])

	hsv[...,0] = ang*180/np.pi/2
	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	
	bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
	x = np.subtract(prev_bgr,bgr)
	r = x[:,:,0].flatten()
	g = x[:,:,1].flatten()
	b = x[:,:,2].flatten()
	
	# if 33 % of pixels any channel change their value by more than abs(10)
	thresh_hold = (x.shape[0] * x.shape[1])//3
	
	t1 = np.count_nonzero(r > 10)
	t2 = np.count_nonzero(g > 10)
	t3 = np.count_nonzero(b > 10)
	
	if(t1 >= thresh_hold or t2 >= thresh_hold or t3 >= thresh_hold):
		return True, bgr
	#print(t1,t2,t3)
	return False , bgr

	
def HSI_Color_Format(img):
	for i in range(img.shape[1]):
		for j in range(img.shape[0]):
			b , g, r = img[j,i]
			sum_bgr = int(int(b)+int(g)+int(r))
			I = sum_bgr/3
			S = 1 - 3*(min([r,g,b])/sum_bgr)
			if(S < 0.00001):
				S = 0
			elif(S > 0.99999):
				S = 1
			rb = int(int(r) - int(b))
			rg = int(int(r) - int(g))
			denom = math.sqrt(( rg * rg) + (rb*(int(g) - int(b))))
			if( S != 0 and denom!=0):
				H = 0.5 * (rg - rb) / math.sqrt(( rg * rg) + (rb*(int(g) - int(b))))
				H = math.acos(H)
				if( b <= g):
					H =H
				else:
					H = 360 - H
			else:
				H = 180
			S = S * 100
			img[j,i] = [I,S,H]
	count = 0
	for i in range(img.shape[1]):
		for j in range(img.shape[0]):
			I ,S , H = img[j,i]
			#Tune values of H,S,I as per demand 
			if (~((H<=120) and (S<=50) and (I>=180))):
				img[j,i] = [0,0,0]
				count+=1
				
	size = (img.shape[1] * img.shape[0])
	flame_per = 100 - (count*100)/size
	
	if flame_per < CON.CONST_FLAME_THRESHOLD:
		#print(flame_per)
		return False
	return True
	
def Flame_Area_Detection(gray):
	thresh = 127
	im_bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
	
	img_linear = im_bw.flatten()
	black_pixel_count = np.count_nonzero(img_linear == 0)
	white_pixel_count = np.count_nonzero(img_linear == 255)
	per_b_pixel = (white_pixel_count/len(img_linear)) * 100
	
	if(per_b_pixel < CON.CONST_AREA_THRESHOLD):
		#print(per_b_pixel)
		return False
	return True

	


cap = cv2.VideoCapture('./../Data/flame.mkv')
ret ,img =  cap.read()

clone = img.copy()
flame_img = clone[CON.CONST_UPPER_LEFT_Y:CON.CONST_BOTTOM_RIGHT_Y , CON.CONST_UPPER_LEFT_X:CON.CONST_BOTTOM_RIGHT_X]

prev = cv2.cvtColor(flame_img , cv2.COLOR_BGR2GRAY)


hsv = np.zeros_like(flame_img)
hsv[...,1] = 255
prev_bgr =hsv


while(True):
	ret, frame = cap.read()
	
	clone = frame.copy()
	next_frame = clone[CON.CONST_UPPER_LEFT_Y:CON.CONST_BOTTOM_RIGHT_Y , CON.CONST_UPPER_LEFT_X:CON.CONST_BOTTOM_RIGHT_X]
	next = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
	
	r1 , prev_bgr = Motion_Detaction(prev , next)
	r2 = HSI_Color_Format(next_frame)
	r3 = Flame_Area_Detection(next)

	if(r1 and r2 and r3):
		cv2.rectangle(frame, (CON.CONST_UPPER_LEFT_X, CON.CONST_UPPER_LEFT_Y), (CON.CONST_BOTTOM_RIGHT_X, CON.CONST_BOTTOM_RIGHT_Y), (0,0,255), 4)
	else:
		#cv2.imwrite("./data/"+str(temp)+'frame.png',frame)
		print("###############################################\n")
		print("Is Frame showing movement :::::::::::::::: ",r1)
		print("Is Flame content valid ::::::::::::::::::: ",r2)
		print("Is Flame size valid :::::::::::::::::::::: ",r3)
		print("\n###############################################\n")
		cv2.rectangle(frame, (CON.CONST_UPPER_LEFT_X, CON.CONST_UPPER_LEFT_Y), (CON.CONST_BOTTOM_RIGHT_X, CON.CONST_BOTTOM_RIGHT_Y), (0,255,0), 4)
	
	cv2.imshow('Flame Detection',frame)
	cv2.imshow('Motion of pixels',prev_bgr)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()