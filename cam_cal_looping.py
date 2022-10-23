import cv2
import numpy as np
import os
import glob
from networktables import NetworkTables

NT = NetworkTables.getTable("ceaseless-watcher")

#Camera Constants
VIDEO_DEV = 2 #Video Device ID for the camera used. Probably 0 or 1 for Webcam, 2 or 3 for internal if on laptop and more than one device
FRAME_HEIGHT = 480 #Height of the camera being used
FRAME_WIDTH = 640 #Width of the camera being used
FRAME_RATE = 30

criteria = (cv2.TERM_CRITERIA_EPS +
			cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

CHECKERBOARD = (7, 7)

# 3D points real world coordinates
objectp3d = np.zeros((1, CHECKERBOARD[0]
					* CHECKERBOARD[1],
					3), np.float32)
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
							0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Vector for 3D points
threedpoints = []

# Vector for 2D points
twodpoints = []

cam = cv2.VideoCapture(VIDEO_DEV) #ID of the camera being used

#Resolution and frame rate settings
cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cam.set(cv2.CAP_PROP_POS_FRAMES, FRAME_RATE)

looping = True

while looping:
    result, image = cam.read()
    grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# Find the chess board corners
	# If desired number of corners are
	# found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(
					grayColor, CHECKERBOARD,
					cv2.CALIB_CB_ADAPTIVE_THRESH
					+ cv2.CALIB_CB_FAST_CHECK +
					cv2.CALIB_CB_NORMALIZE_IMAGE)
	# If desired number of corners can be detected then,
	# refine the pixel coordinates and display
	# them on the images of checker board
    if ret == True:
        threedpoints.append(objectp3d)
		# Refining pixel coordinates
		# for given 2d points.
        corners2 = cv2.cornerSubPix(
			grayColor, corners, (11, 11), (-1, -1), criteria)

        twodpoints.append(corners2)

		# Draw and display the corners
        image = cv2.drawChessboardCorners(image,
										CHECKERBOARD,
										corners2, ret)



    else:
        print("No corners found")
    cv2.imshow('img', image)

    key = cv2.waitKey(100)

    if key == 13:
        looping = False

    #cameraServer() #Uncomment if you want things to break

cv2.destroyAllWindows()
cv2.imwrite("final.png", image)


h, w = image.shape[:2]

# Perform camera calibration by
# passing the value of above found out 3D points (threedpoints)
# and its corresponding pixel coordinates of the
# detected corners (twodpoints)
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
    threedpoints, twodpoints, grayColor.shape[::-1], None, None)

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w,h), 1, (w,h))

# Displaying required output
print(" Camera matrix:")
print(matrix)

print("\n Distortion coefficient:")
print(distortion)

print("\n Rotation Vectors:")
print(r_vecs)

print("\n Translation Vectors:")
print(t_vecs)

#[[768.2033943    0.         296.90924324]
#[  0.         793.22074413 254.28872497]
#[  0.           0.           1.        ]]

#Distortion coefficient:
#[[-1.03006422e-01  4.12686114e-01  4.85352291e-04 -1.75570638e-03
# -1.04181606e+00]]



dst = cv2.undistort(image, matrix, distortion, None, newcameramtx)
# crop the image

cv2.imwrite('calibresult.png', dst)

# Camera matrix:
#[[755.90120825   0.         313.55829919]
#[  0.         787.42158023 238.84075425]
#[  0.           0.           1.        ]]

#Distortion coefficient:
#[[-0.13852351  0.24669332  0.00165809  0.00048304 -0.2890661 ]]

mean_error = 0
for i in range(len(threedpoints)):
    imgpoints2, _ = cv2.projectPoints(threedpoints[i], r_vecs[i], t_vecs[i], matrix, distortion)
    error = cv2.norm(twodpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(threedpoints)) )


# Camera matrix:
#[[2.79351724e+03 0.00000000e+00 3.37621619e+02]
#[0.00000000e+00 2.88140551e+03 2.65131081e+02]
#[0.00000000e+00 0.00000000e+00 1.00000000e+00]]

#Distortion coefficient:
#[[-2.45326809e+00  1.28464799e+02  2.88828541e-02 -4.27263376e-02
# -3.74878188e+03]]
#