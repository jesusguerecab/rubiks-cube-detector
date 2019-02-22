# Simple bayesian correlation filter
# Programmed by Olac Fuentes
# Last modified November 19, 2018

import numpy as np
import cv2
import time

def get_coordinates(event,x,y,flags,param):
    global x0, x1, y0, y1
    if event == cv2.EVENT_LBUTTONDOWN:
        if x0==-1:
            x0=x
            y0=y
            print('Entered x0 =',x0,'y0 =',y0)
        elif x1==-1:
            x1=x+1
            y1=y+1
            print('Entered x1 =',x1,'y1 =',y1)

def get_pattern():
    global x0, x1, y0, y1
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',get_coordinates)
    while(x1==-1):
        ret, frame = cap.read()
        #frame = cv2.GaussianBlur(frame,(5,5),0)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255
        kernel_v = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        kernel_h = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        np.array([[-1,-1],[1,1]])
        img = np.abs(cv2.filter2D(gray_frame,-1,kernel_v))+np.abs(cv2.filter2D(gray_frame,-1,kernel_h))
        img, img2 = cv2.threshold(img, np.mean(img), 255, cv2.THRESH_BINARY)
        
        cv2.imshow('image',img2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow('image')
    return (img2[y0:y1,x0:x1]).astype(np.float)

x0=x1=y0=y1=-1
cap = cv2.VideoCapture(0)
pattern = get_pattern()
mean_filt = np.ones((pattern.shape[0],pattern.shape[1]))
mean_filt = mean_filt/np.sum(mean_filt)
py = pattern.shape[0]
px = pattern.shape[1]
cv2.imshow('pattern',pattern)
for i in range(3):
    pattern[:,:] = pattern[:,:] - np.mean(pattern[:,:])

start = time.time()
count=0
rows = 480
cols = 640
fm = np.zeros((rows,cols))
col_mat = np.tile(np.arange(cols),(rows,1))
row_mat = np.tile(np.arange(rows),(cols,1)).T
k=1000

while(True):
    count+=1
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)/255
    #frame = cv2.GaussianBlur(frame,(5,5),0)
    kernel_v = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    kernel_h = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    np.array([[-1,-1],[1,1]])
    frame = np.abs(cv2.filter2D(gray_frame,-1,kernel_v))+np.abs(cv2.filter2D(gray_frame,-1,kernel_h))
    img, frame = cv2.threshold(frame, np.float32(np.mean(frame)), 255, cv2.THRESH_BINARY)

    fm[:,:]=cv2.filter2D(frame[:,:],-1,mean_filt,anchor =(0,0))
    centered_frame = frame - fm

    match=cv2.filter2D(centered_frame[:,:],-1,pattern[:,:])
    match = match/np.max(match)

    #prior_x = col_mat - frame.shape[1]/2 - px/2
    #prior_y = row_mat - frame.shape[0]/2 - py/2
    prior_x = col_mat - x0
    prior_y = row_mat - y0
    prior = prior_x*prior_x +prior_y*prior_y
    prior = np.exp(-prior/k)
    combined = match*prior
    cv2.imshow('prior probability',prior)
    cv2.imshow('match probability',match)
    cv2.imshow('combined probability',combined)
    y0,x0 = np.unravel_index(combined.argmax(), match.shape)

    cv2.rectangle(frame,(x0-px//2, y0-py//2), (x0+px//2, y0+py//2),(255,255,255),1)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

elapsed_time = time.time()-start
print('Capture speed: {0:.2f} frames per second'.format(count/elapsed_time))

cap.release()
cv2.destroyAllWindows()
