#Import the necessay packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np

#Initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640,480))

#Allow the camera to warmup
time.sleep(0.1)

#Capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    #Grab the raw NumPy array representing the image, then initialize the timestamp
    #and occupied/unoccupied text
    image = frame.array
    
    #Note: In case of the need to flip the video, enable the code below
    #image = cv2.flip(image,1)
    
    #Convert video from BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #Define the lower and upper HSV values for red. Then create the mask for red
    #The mask is a threshold to only get red.
    lower_red = np.array([161,155,84])
    upper_red = np.array([179,255,255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    
    #Define the lower and upper HSV values for yellow. Then create the mask for yellow
    #The mask is a threshold to only get yellow.
    lower_yellow = np.array([20,100,100])
    upper_yellow = np.array([45,255,255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    #Define the lower and upper HSV values for green. Then create the mask for green
    #The mask is a threshold to only get green.
    lower_green = np.array([50,54,60])
    upper_green = np.array([80,255,255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    #Combine all masks
    mask = mask_red + mask_yellow + mask_green
    
    #Bitwise-AND the image and mask
    res = cv2.bitwise_and(image, image, mask = mask)

    #Show the original, mask, and results
    cv2.imshow("Original", image)
    cv2.imshow("Mask", mask)
    cv2.imshow("Results", res)
    key = cv2.waitKey(1) & 0xFF
    
    #clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    
    #if the 'q' key is pressed, break from loop
    if key == ord("q"):
        break
