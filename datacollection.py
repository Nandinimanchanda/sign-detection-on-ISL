import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np 
import math 
import time
cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)
detector = HandDetector(maxHands=2)
offset = 20
imgSize= 500
folder="C:\\Users\\nandi\\.vscode\\ascii_art\\sign detection\\data\\A\\v"
counter = 0
while True:
    success, img = cap.read()
    hands,img = detector.findHands(img)
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
        else:
            k = imgSize/w
            hCal = math.ceil(k * h)
            
            imgResize=cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap,:] = imgResize
        cv2.imshow("imageCrop",imgCrop)
        cv2.imshow("imageWhite",imgWhite)
    cv2.imshow("image",img)
    key = cv2.waitKey(8)
    if key == ord("s"):
        counter +=1
        cv2.imwrite(f'{folder}/image_{time.time()}.jpg',imgWhite)
        print(counter)
  