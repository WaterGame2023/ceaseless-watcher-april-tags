import math
import cv2
import numpy as np
import apriltag 
from networktables import NetworkTables
#from cscore import CameraServer

#NetworkTables.initialize(server='10.25.31.2') #Uncomment when there is a NT server
NT = NetworkTables.getTable("ceaseless-watcher")

LINE_LENGTH = 5
CENTER_COLOR = (0, 255, 0)
CORNER_COLOR = (255, 0, 255)

#Camera Constants
VIDEO_DEV = 2 #Video Device ID for the camera used. Probably 0 or 1 for Webcam, 2 or 3 for internal if on laptop and more than one device
FRAME_HEIGHT = 480 #Height of the camera being used
FRAME_WIDTH = 640 #Width of the camera being used
FRAME_RATE = 60 #Frame rate of the camera being used

#def cameraServer():
 #   camServe = CameraServer.getInstance()
  #  camServe.enableLogging()
    #camServe.startAutomaticCapture(image) #this might work
   # camServe.waitForever()

TAG_SIZE = .2 #Tag size in meters

camInfo = np.matrix([[2.79351724e+03, 0.00000000e+00, 3.37621619e+02], [0.00000000e+00, 2.88140551e+03, 2.65131081e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00] ])
distCoeffs = np.matrix([[-2.45326809e+00], [1.28464799e+02], [2.88828541e-02], [-4.27263376e-02], [-3.74878188e+03]])

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

options = apriltag.DetectorOptions(families="all")
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
        print("No tags detected")
    else:
        for detect in detections:
            centerX = detect.center[0]
            centerY = detect.center[1]

            centerOriginX = (centerX - (FRAME_WIDTH / 2))
            centerOriginY = ((FRAME_HEIGHT / 2) - centerY)

            print("\nX-Axis:", centerOriginX, "\n") #Debug
            print("Y-Axis:", centerOriginY, "\n") #Debug

            image = plotPoint(image, detect.center, CENTER_COLOR)
            image = plotText(image, detect.center, CENTER_COLOR, detect.tag_id)

            for corner in detect.corners:
                image = plotPoint(image, corner, CORNER_COLOR)

            varName = TAG_SIZE/2
            objectPoints= np.array([ [-varName,varName, 0], [ varName, varName, 0], [ varName, -varName, 0], [-varName, -varName, 0] ])
            SOLVEPNP_IPPE_SQUARE =7 # (enumeration not working: 
            # https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga357634492a94efe8858d0ce1509da869)
                
            #print(d['lb-rb-rt-lt'])
            imagePoints = np.array([detect.corners])
            #print(imagePoints)
            # solvePnP docs: https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d
            retval, tvec, rvec = cv2.solvePnP(objectPoints, imagePoints, camInfo, None, useExtrinsicGuess=True, flags=SOLVEPNP_IPPE_SQUARE)
            #print(cv2.solvePnP(objectPoints, imagePoints, camInfo, None, useExtrinsicGuess=False, flags=SOLVEPNP_IPPE_SQUARE))
            #print("rvec:", rvec)
            #print("tvec:", tvec)
            R = cv2.Rodrigues(rvec)[0]
            # print("R:", R)
            yaw = np.arctan2(R[0,2],R[2,2])*(180/np.pi) # 180//np.pi gets to integers?
            roll = np.arcsin(-R[1][2])*(180/np.pi)
            pitch = np.arctan2(R[1,0],R[1,1])*(180/np.pi)

            yrpText = "Y: " + str(round(yaw, 2)) + " R: " + str(round(roll, 2)) + " P: " + str(round(pitch, 2))
            image = plotText(image, (detect.center[0], detect.center[1] + 20), CENTER_COLOR, yrpText)


    dst = cv2.undistort(image, camInfo, distCoeffs, None, camInfo)
    cv2.imshow('Vid-Stream', dst) #Comment out when running in headless mode to not piss off python

    key = cv2.waitKey(100)

    if key == 13:
        looping = False

    #cameraServer() #Uncomment if you want things to break

cv2.destroyAllWindows()
cv2.imwrite("final.png", image)
