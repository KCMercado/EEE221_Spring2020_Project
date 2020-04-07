#Import the necessay packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np

#Initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.rotation = 180 #rotates camera by 180 degrees b/c of camera setup
camera.resolution = (640,480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640,480))
size = camera.shape

#Allow the camera to warmup
time.sleep(0.1)

#Define the codec and create Videowriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('recording_test.avi',fourcc,5.0,(640,480))

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
    #The mask is a threshold to only get red. However there are two set of range for red
    lower_red1 = np.array([0,100,100])
    upper_red1 = np.array([10,255,255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([160,100,100])
    upper_red2 = np.array([180,255,255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    mask_redComb = mask_red1 + mask_red2
    
    #Define the lower and upper HSV values for yellow. Then create the mask for yellow
    #The mask is a threshold to only get yellow.
    lower_yellow = np.array([15,150,150])
    upper_yellow = np.array([35,255,255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    #Define the lower and upper HSV values for green. Then create the mask for green
    #The mask is a threshold to only get green.
    lower_green = np.array([40,50,50])
    upper_green = np.array([90,255,255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    #Combine all masks
##    mask = mask_red1 + mask_red2 + mask_yellow + mask_green
    
    #Hough Transform circle detection for all three colors
    r_circles = cv2.HoughCircles(mask_redComb, cv2.HOUGH_GRADIENT, 1, 80, param1=50, param2=10, minRadius=0, maxRadius=30)
    y_circles = cv2.HoughCircles(mask_yellow, cv2.HOUGH_GRADIENT, 1, 30, param1=50, param2=5, minRadius=0, maxRadius=30)
    g_circles = cv2.HoughCircles(mask_green, cv2.HOUGH_GRADIENT, 1, 60, param1=50, param2=10, minRadius=0, maxRadius=30)
    
    #Traffic light detect
    r = 5
    bound = 4.0/10
    
    #
    if r_circles is not None:
        r_circles = np.uint16(np.around(r_circles))
        
        for i in r_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                continue
            
    
    #Perform morphological operations to remove any small blobs (noise)
##    mask = cv2.erode(mask, None, iterations=2)
##    mask = cv2.dilate(mask, None, iterations=2)

    #Bitwise-AND the image and mask
    res = cv2.bitwise_and(image, image, mask = mask)

    #Create contours of objects
##    (contours, _) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
##    contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)
##    for cnt in contours:
##        (x, y, w, h) = cv2.boundingRect(cnt)
##        cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0),2)
##        cv2.rectangle(res, (x, y), (x+w, y+h), (0,255,0),2)

    #Write to the out file
    out.write(image)
    
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
