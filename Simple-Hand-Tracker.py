#Import the necessary Packages for this software to run
import mediapipe
import cv2
from collections import Counter
from module import findnameoflandmark, findpostion, speak
import math

import os
from time import *
import RPi.GPIO as GPIO
GPIO.setwarnings(False)

#Use MediaPipe to draw the hand framework over the top of hands it identifies in Real-Time
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands

#Use CV2 Functionality to create a Video stream and add some values
cap = cv2.VideoCapture(0)
tip=[8,12,16,20]
tipname=[8,12,16,20]
fingers=[]
finger=[]

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)

#Add confidence values and extra settings to MediaPipe hand tracking. As we are using a live video stream this is not a static
#image mode, confidence values in regards to overall detection and tracking and we will only let two hands be tracked at the same time
#More hands can be tracked at the same time if desired but will slow down the system
with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:

#Create an infinite loop which will produce the live feed to our desktop and that will search for hands
     while True:
           ret, frame = cap.read()
           #Unedit the below line if your live feed is produced upsidedown
           #flipped = cv2.flip(frame, flipCode = -1)
           
           #Determines the frame size, 640 x 480 offers a nice balance between speed and accurate identification
           frame1 = cv2.resize(frame, (640, 480))
           
           frame1 = cv2.flip(frame1, 1)
           
           a=findpostion(frame1)
           b=findnameoflandmark(frame1)
           
           if len(b and a)!=0:
            finger=[]
            if a[0][1:] < a[4][1:]: 
               finger.append(1)
               print (b[4])
          
            else:
               finger.append(0)   
        
            fingers=[] 
            for id in range(0,4):
                if a[tip[id]][2:] < a[tip[id]-2][2:]:
                   print(b[tipname[id]])

                   fingers.append(1)
    
                else:
                   fingers.append(0)
           #Below will print to the terminal the number of fingers that are up or down          
           x=fingers + finger
           c=Counter(x)
           up=c[1]
           down=c[0]
           print('This many fingers are up - ', up)
           print('This many fingers are down - ', down)
        
           #produces the hand framework overlay ontop of the hand, you can choose the colour here too)
           results = hands.process(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
           
           #Incase the system sees multiple hands this if statment deals with that and produces another hand overlay
           if results.multi_hand_landmarks != None:
              for handLandmarks in results.multi_hand_landmarks:
                  drawingModule.draw_landmarks(frame1, handLandmarks, handsModule.HAND_CONNECTIONS)
                  
                  #   Added Code to find Location of Index Finger !!
                  #for point in handsModule.HandLandmark:
                      
                      #normalizedLandmark = handLandmarks.landmark[point]
                      #pixelCoordinatesLandmark= drawingModule._normalized_to_pixel_coordinates(normalizedLandmark.x, normalizedLandmark.y, 640, 480)
                      
                      #if point == 8:
                          #print(point)
                          #print(pixelCoordinatesLandmark)
                          #print(normalizedLandmark)
            
           #Below shows the current frame to the desktop 
           cv2.imshow("Frame", frame1);
           key = cv2.waitKey(1) & 0xFF
           
           if up == 1:
               GPIO.output(17, GPIO.LOW)
           
           if up == 2:
               GPIO.output(17, GPIO.HIGH)
               
           
           #Below states that if the |q| is press on the keyboard it will stop the system
           if key == ord("q"):
              break

