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
VIDEO_DEV = 2 #Video Device ID for the camera used. Probably 0 or 1 for Webcam, 2 or 3 for internal if on laptop and more than one device
FRAME_HEIGHT = 480 #Height of the camera being used
FRAME_WIDTH = 640 #Width of the camera being used
FRAME_RATE = 30

def cameraServer():
    camServe = CameraServer.getInstance()
    camServe.enableLogging()
    camServe.startAutomaticCapture(image) #this might work
    camServe.waitForever()

TAG_SIZE = .2 #Tag size in meters

camInfo = np.matrix([ [1430, 0, 320], [ 0, 1430, 240], [ 0, 0, 1] ])

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

def plotText(image, center, color, text):
    center = (int(center[0]) + 4, int(center[1]) - 4)
    return cv2.putText(image, str(text), center, cv2.FONT_HERSHEY_SIMPLEX,
                       1, color, 3)

detector = apriltag.Detector()
cam = cv2.VideoCapture(VIDEO_DEV) #ID of the camera being used

#Resolution and frame rate settings
cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cam.set(cv2.CAP_PROP_POS_FRAMES, FRAME_RATE)

looping = True

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
            print("\ntag-id: %s center-x: %s \ntag-id: %s center-y: %s" % (detect.tag_id, detect.center[1], detect.tag_id, detect.center[0]))

            if detect.tag_id == 69:
                print("UwU 💖💖✨🥺,,👉👈💖💖✨🥺,,,,👉👈💖💖✨🥺,,👉👈✨✨✨,,👉👈💖💖✨🥺👉👈💖💖✨🥺,,,,👉👈💖💖,👉👈💖💖✨✨👉👈💖💖✨✨,👉👈✨✨✨,,👉👈💖💖✨,,,,👉👈💖💖✨,👉👈💖💖✨🥺,,,,👉👈💖💖✨,👉👈💖✨✨✨✨🥺,,,👉👈💖💖✨,👉👈💖💖✨🥺,👉👈")

            centerX = detect.center[0]
            centerY = detect.center[1]

            centerOriginX = (centerX - (FRAME_WIDTH / 2))
            centerOriginY = ((FRAME_HEIGHT / 2) - centerY)

            #print("\nX-Axis:", centerOriginX, "\n") #Debug
            #print("Y-Axis:", centerOriginY, "\n") #Debug

            #print("\ntag-id:", detect.tag_id, "center-x:", centerX) #Debug
            #print("tag-id:", detect.tag_id, "center-y:", centerY) #Debug

            image = plotPoint(image, detect.center, CENTER_COLOR)
            image = plotText(image, detect.center, CENTER_COLOR, detect.tag_id)

            NT.putString("tag_center", detect.center) #Uses default openCV Coordinate system w/ origin top-left
            NT.putString("tag_x", centerOriginX) #x-axis value of tag
            NT.putString("tag_y", centerOriginY) #y-axis value of tag
            NT.putString("tag_id", detect.tag_id)
            NT.putString("tagfound", 1)

            for corner in detect.corners:
                image = plotPoint(image, corner, CORNER_COLOR)

            varName = TAG_SIZE/2
            objectPoints= np.array([ [-varName,varName, 0], [ varName, varName, 0], [ varName, -varName, 0], [-varName, -varName, 0] ])
            SOLVEPNP_IPPE_SQUARE =7 # (enumeration not working: 
            # https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga357634492a94efe8858d0ce1509da869)

        for d in detections:
                
            #print(d['lb-rb-rt-lt'])
            imagePoints = np.array([detect.corners])
            #print(imagePoints)
            # solvePnP docs: https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d
            retval, tvec, rvec = cv2.solvePnP(objectPoints, imagePoints, camInfo, None, useExtrinsicGuess=False, flags=SOLVEPNP_IPPE_SQUARE)
            #print(cv2.solvePnP(objectPoints, imagePoints, camInfo, None, useExtrinsicGuess=False, flags=SOLVEPNP_IPPE_SQUARE))
            #print("rvec:", rvec)
            #print("tvec:", tvec)
            #R = cv2.Rodrigues(rvec)[0]
            #print("R:", R)
            #yaw = np.arctan2(R[0,2],R[2,2])*180/np.pi # 180//np.pi gets to integers?
            #roll = np.arcsin(-R[1][2])*180/np.pi
            #pitch = np.arctan2(R[1,0],R[1,1])*180/np.pi

            # rvec_matrix = cv2.Rodrigues(rvec)[0]
            # proj_matrix = np.hstack((rvec_matrix, tvec))
            # eulerAngles = -cv2.decomposeProjectionMatrix(proj_matrix)[6] 

            # yaw_degrees   = eulerAngles[1]
            # pitch_degrees = eulerAngles[0]
            # roll_degrees  = eulerAngles[2]

            # print("\n", yaw_degrees)
            # print(pitch_degrees)
            # print(roll_degrees)

    cv2.imshow('Vid-Stream', image) #Comment out when running in headless mode to not piss off python

    key = cv2.waitKey(100)

    if key == 13:
        looping = False

    #cameraServer() #Uncomment if you want things to break

cv2.destroyAllWindows()
cv2.imwrite("final.png", image)
