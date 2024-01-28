import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np 
import math 
import time
import tensorflow 
cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)
detector = HandDetector(maxHands=2)
classifier = Classifier("C:\\Users\\nandi\\.vscode\\ascii_art\\sign detection\\model\\keras_model.h5","C:\\Users\\nandi\\.vscode\\ascii_art\\sign detection\\model\\labels.txt")
offset = 20
imgSize=500
folder="C:\\Users\\nandi\\.vscode\\ascii_art\\sign detection\\data\\A\\v"
counter = 0
labels = ["v","c"]
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands,img = detector.findHands(img  ,flipType=False)
    if hands:
        hand = hands[0]
        x,y,w,h=hand['bbox']
        
        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255
        imgCrop = img[y-offset:y+h+offset,x-offset:x+w+offset]
        
        imgCropShape = imgCrop.shape
    
        
        aspectRatio = h/w
        
        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k * w)
            
    
            imgResize=cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:,wGap:wCal+wGap] = imgResize
            prediction,index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction,index)
            # cv2.putText(imgOutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        else:
            k = imgSize/w
            hCal = math.ceil(k * h)
            
            imgResize=cv2.resize(imgCrop,(hCal,imgSize))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap,:] = imgResize
            prediction,index = classifier.getPrediction(imgWhite,draw=False)
        cv2.rectangle(imgOutput,(x-offset,y-offset-50),(x-offset+90,y-offset-50+50),(255,0,255),cv2.FILLED)
        cv2.putText(imgOutput,labels[index],(x,y-26),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),4)
        cv2.imshow("imageCrop",imgCrop)
        cv2.imshow("imageWhite",imgWhite)
    cv2.imshow("image",imgOutput)
    cv2.waitKey(8)
    # if key == ord("s"):
    #     counter +=1
    #     cv2.imwrite(f'{folder}/image_{time.time()}.jpg',imgWhite)
    #     print(counter)
        