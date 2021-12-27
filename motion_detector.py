import cv2
import numpy as np
import pandas as pd

Frame_1=None
cam=cv2.VideoCapture(0)

while True:
    success,frame=cam.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(5,5),0)
    if Frame_1 is None:
        Frame_1=gray
        continue
    delta_frame=cv2.absdiff(Frame_1,gray)
    thresh_delta=cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
    thresh_delta=cv2.dilate(thresh_delta,None,iterations=0)
    (cnts,_)=cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts:
        if cv2.contourArea(contour)<1000:
            continue
        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow('Frame',frame)
    # cv2.imshow('Capturing',gray)
    # cv2.imshow('delta',delta_frame)
    # cv2.imshow('thresh',thresh_delta)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()



