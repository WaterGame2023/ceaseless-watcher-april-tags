from binhex import LINELEN
import math
from tkinter import NONE
import cv2
import numpy as np
import apriltag
from networktables import NetworkTables
from cscore import CameraServer

#NetworkTables.initialize(server='10.25.31.2') #Uncomment when there is a NT server
NT = NetworkTables.getTable("ceaseless-watcher")

LINE_LENGTH = 5
CENTER_COLOR = (0, 255, 0)
CORNER_COLOR = (255, 0, 255)

#Camera Constants
VIDEO_DEV = 0 #Video Device ID for the camera used. Probably 0 or 1 for Webcam, 2 or 3 for internal if on laptop and more than one device
FRAME_HEIGHT = 480 #Height of the camera being used
FRAME_WIDTH = 640 #Width of the camera being used
FRAME_RATE = 30 #Desired Frame Rate

#Starts camera server
def cameraServer():
    camServe = CameraServer.getInstance()
    camServe.enableLogging()
    camServe.startAutomaticCapture(image) #this might work
    camServe.waitForever()

TAG_SIZE = .2 #Tag size in meters

#Camera Information thats needed for solvePnp
camInfo = np.matrix([[689.86477877,   0,         312.77834974],
 [  0,         695.01487988, 280.708403  ],
 [  0,           0,           1.0        ]]) #FIXME For when you use this you will need to run camera_calib unless you want stuff to be really wrong

#Distsortion Coefficients for the camera being used
distCoeff = np.matrix([[-5.32044320e-02,  4.61488555e-01, -9.37542209e-04, -2.00168792e-03, -1.30772959e+00]]) #FIXME For when you use this you will need to run camera_calib unless you want stuff to be really wrong

#Plots points and draws lines for the corner of the tags
def plotPoint(image, center, color):
    center = (int(center[0]), int(center[1]))
    image = cv2.line(image,
                     (center[0] - LINE_LENGTH, center[1]),
                     (center[0] + LINE_LENGTH, center[1]),
                     color,
                     3)
    image = cv2.line(image,
                     (center[0], center[1] - LINE_LENGTH),
                     (center[0], center[1] + LINE_LENGTH),
                     color,
                     3)
    return image

#Plots the tag ID in the center of the tag
def plotText(image, center, color, text):
    center = (int(center[0]) + 4, int(center[1]) - 4)
    return cv2.putText(image, str(text), center, cv2.FONT_HERSHEY_SIMPLEX,
                       1, color, 3)

#Initializes the AprilTag detector and sets the Camera ID
detector = apriltag.Detector()
cam = cv2.VideoCapture(VIDEO_DEV) #ID of the camera being used

#Sets the resolution and frame rate settings
cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cam.set(cv2.CAP_PROP_POS_FRAMES, FRAME_RATE)

looping = True #Starts looping the fun stuff

while looping:
    result, image = cam.read()
    grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# look for tags
    detections = detector.detect(grayimg)
    if not detections:
        NT.putString("tagfound", 0)
        print("No Tag found.  Looking for tags")
    else:
        for detect in detections:
            #print("\ntag_id: %s, center-yx: %s" % (detect.tag_id, detect.center))
            print("\ntag-id: %s center-x: %s \ntag-id: %s center-y: %s" % (detect.tag_id, detect.center[1], detect.tag_id, detect.center[0])) #Prints the tag ID and the center coordinates of the tag

            #Really stupid thing to output if the tag ID is 69
            if detect.tag_id == 69:
                print("UwU 💖💖✨🥺,,👉👈💖💖✨🥺,,,,👉👈💖💖✨🥺,,👉👈✨✨✨,,👉👈💖💖✨🥺👉👈💖💖✨🥺,,,,👉👈💖💖,👉👈💖💖✨✨👉👈💖💖✨✨,👉👈✨✨✨,,👉👈💖💖✨,,,,👉👈💖💖✨,👉👈💖💖✨🥺,,,,👉👈💖💖✨,👉👈💖✨✨✨✨🥺,,,👉👈💖💖✨,👉👈💖💖✨🥺,👉👈")

            #Detects the center of the tag and outputs the X and Y coordinates individually
            centerX = detect.center[0]
            centerY = detect.center[1]

            #Makes the X and Y coordinates relative to the center of the frame
            centerOriginX = (centerX - (FRAME_WIDTH / 2))
            centerOriginY = ((FRAME_HEIGHT / 2) - centerY)

            #Debug stuff for outputting the center of the tag
            #print("\nX-Axis:", centerOriginX, "\n") #Debug
            #print("Y-Axis:", centerOriginY, "\n") #Debug

            #print("\ntag-id:", detect.tag_id, "center-x:", centerX) #Debug
            #print("tag-id:", detect.tag_id, "center-y:", centerY) #Debug

            #Plots the tag ID and a cross-hair in the center of the tag
            image = plotPoint(image, detect.center, CENTER_COLOR)
            image = plotText(image, detect.center, CENTER_COLOR, detect.tag_id)

            #Outputs the tag ID and the center coordinates of the tag to NetworkTables
            NT.putString("tag_center", detect.center) #Uses default openCV Coordinate system w/ origin top-left
            NT.putString("tag_x", centerOriginX) #x-axis value of tag
            NT.putString("tag_y", centerOriginY) #y-axis value of tag
            NT.putString("tag_id", detect.tag_id)
            NT.putString("tagfound", 1)

            #Plots points in the corners of the tag
            for corner in detect.corners:
                image = plotPoint(image, corner, CORNER_COLOR)

        #Stuff needed for SolvePnP
        halfTagSize = TAG_SIZE/2
        objectPoints= np.array([ [-halfTagSize,halfTagSize, 0], [ halfTagSize, halfTagSize, 0], [ halfTagSize, -halfTagSize, 0], [-halfTagSize, -halfTagSize, 0] ])
        SOLVEPNP_IPPE_SQUARE =7 # (enumeration not working: 
        # https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga357634492a94efe8858d0ce1509da869)

        for d in detections:
                
            #print(d['lb-rb-rt-lt']) #Debug
            imagePoints = np.array([detect.corners]) #Outputs corners of tag as array
            #print(imagePoints) #Debug

            # solvePnP docs: https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d

            #solvePNP returns rotation and translation vectors
            retval, tvec, rvec = cv2.solvePnP(objectPoints, imagePoints, camInfo, distCoeff, useExtrinsicGuess=False, flags=SOLVEPNP_IPPE_SQUARE)
            #print(cv2.solvePnP(objectPoints, imagePoints, camInfo, distCoeff, useExtrinsicGuess=False, flags=SOLVEPNP_IPPE_SQUARE))
            #print("rvec:", rvec)
            #print("tvec:", tvec)
            R = cv2.Rodrigues(tvec)[0]
            # print("R:", R)

            #Convert to yaw, pitch, roll in degrees
            # yaw = (np.arctan2(R[0,2],R[2,2])*180/np.pi) % 180 # 180//np.pi gets to integers?
            # pitch = np.arcsin(-R[1][2])*180/np.pi
            # roll = (np.arctan2(R[1,0],R[1,1])*180/np.pi) + 180

            #New stuff for debug
            sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0]) #Idk what this does tbh
            yaw = math.atan2(-R[1,0], R[0,0]) #Debug
            pitch = math.atan2(-R[2,0], sy) #Debug
            roll = math.math.atan2(R[2,1] , R[2,2]) #Debug

            #This stuff only outputs in euler angles and should probably be removed but am keeping for debugging purposes
            # rvec_matrix = cv2.Rodrigues(rvec)[0] #Debug
            # proj_matrix = np.hstack((rvec_matrix, tvec)) #Debug
            # eulerAngles = -cv2.decomposeProjectionMatrix(proj_matrix)[6] #Debug
            # yaw   = eulerAngles[1] #Debug
            # pitch = eulerAngles[0] #Debug
            # roll  = eulerAngles[2] #Debug

            #Output yaw, pitch, roll values to command line
            print("\nYaw", yaw)
            print("pitch", pitch)
            print("roll", roll)

            #Output yaw, pitch, roll values to NetworkTables
            NT.putString("yaw", yaw)
            NT.putString("pitch", pitch)
            NT.putString("roll", roll)

    #Output window with the live feed from the camera and overlays
    cv2.imshow('Vid-Stream', image) #Comment out when running in headless mode to not piss off python

    #Defines enter key and a 100ms delay before exiting the program
    key = cv2.waitKey(100)

    #If the enter key is pressed exit the program
    if key == 13:
        looping = False

    #cameraServer() #Uncomment if you want things to break

#Closes the output window and writes the output file
cv2.destroyAllWindows()
cv2.imwrite("final.png", image)
