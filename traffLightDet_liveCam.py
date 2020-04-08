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

#Allow the camera to warmup
time.sleep(0.1)

#Define the codec and create Videowriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('traffLightDet_recording.mp4',fourcc,5.0,(640,480))

#Define font style
font = cv2.FONT_HERSHEY_SIMPLEX

#Capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):

    #Grab the raw NumPy array representing the image, then initialize the timestamp
    #and occupied/unoccupied text
    image = frame.array
    
    #Note: In case of the need to virtically flip the video, enable the code below
    #image = cv2.flip(image,1)
    
    #Convert video from BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #Define the lower and upper HSV values for red. Then create the mask for red
    #The mask is a threshold to only get red. However there are two set of ranges for red
    lower_red1 = np.array([0,180,180])
    upper_red1 = np.array([8,255,255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170,150,150])
    upper_red2 = np.array([180,255,255])
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)

    #Combine red masks
    mask_redComb = mask_red1 + mask_red2
    
    #Define the lower and upper HSV values for yellow. Then create the mask for yellow
    #The mask is a threshold to only get yellow.
    lower_yellow = np.array([15,150,150])
    upper_yellow = np.array([35,255,255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    #Define the lower and upper HSV values for green. Then create the mask for green
    #The mask is a threshold to only get green.
    lower_green = np.array([55,111,111])
    upper_green = np.array([100,255,255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    #Perform morphological operations to remove any small blobs
    mask_redComb = cv2.erode(mask_redComb, None, iterations=1)
    mask_redComb = cv2.dilate(mask_redComb, None, iterations=1)

    mask_yellow = cv2.erode(mask_yellow, None, iterations=1)
    mask_yellow = cv2.dilate(mask_yellow, None, iterations=1)

    mask_green = cv2.erode(mask_green, None, iterations=1)
    mask_green = cv2.dilate(mask_green, None, iterations=1)
    
    #Combine all masks
    mask = mask_redComb + mask_yellow + mask_green
    
    #Bitwise-AND the image and mask
    res = cv2.bitwise_and(image, image, mask = mask)

    #Hough transform circle detection for all three colors
    r_circles = cv2.HoughCircles(mask_redComb, cv2.HOUGH_GRADIENT, 1, 80, param1=50, param2=8, minRadius=0, maxRadius=50)
    y_circles = cv2.HoughCircles(mask_yellow, cv2.HOUGH_GRADIENT, 1, 80, param1=50, param2=8, minRadius=0, maxRadius=50)
    g_circles = cv2.HoughCircles(mask_green, cv2.HOUGH_GRADIENT, 1, 60, param1=50, param2=10, minRadius=0, maxRadius=50)
    
    #Label and identify red circles
    if r_circles is not None:
        #Convert the (x,y) coordinates and radius of the circles to integers
        r_circles = np.uint16(np.around(r_circles))

        for i in r_circles[0,:]:
            #Draw the outer circle & label
            cv2.circle(image,(i[0],i[1]),i[2],(255,0,0),1)
            cv2.putText(image, 'Red', (i[0],i[1]), font, 1, (255,0,0), 1, cv2.LINE_AA)

    #Label and identify yellow circles
    if y_circles is not None:
        #Convert the (x,y) coordinates and radius of the circles to integers
        y_circles = np.uint16(np.around(y_circles))
    
        for i in y_circles[0,:]:
            #Draw the outer circle & label
            cv2.circle(image,(i[0],i[1]),i[2],(225,0,0),1)
            cv2.putText(image, 'Yellow', (i[0],i[1]), font, 1, (255,0,0), 1, cv2.LINE_AA)

    #Label and identify green circles
    if g_circles is not None:
        #Convert the (x,y) coordinates and radius of the circles to integers
        g_circles = np.uint16(np.around(g_circles))
    
        for i in g_circles[0,:]:
            #Draw the outer circle & label
            cv2.circle(image,(i[0],i[1]),i[2],(255,0,0),1)
            cv2.putText(image, 'Green', (i[0],i[1]), font, 1, (255,0,0), 1, cv2.LINE_AA)
    
    #Show the original, masks, and results
    cv2.imshow("Original", image)
    cv2.imshow("Mask Red", mask_redComb)
    cv2.imshow("Mask Yellow", mask_yellow)
    cv2.imshow("Mask Green", mask_green)
    cv2.imshow("Mask All", mask)
    cv2.imshow("Results", res)


    #Write to the out file (this saves a video)
    out.write(image)
    
    key = cv2.waitKey(1) & 0xFF
    
    #clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    
    #if the 'q' key is pressed, break from loop
    if key == ord("q"):
        break
