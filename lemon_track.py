from imutils.video import VideoStream
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import numpy as np
import argparse
import cv2
import imutils
import math

# add argument for choosing video file
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
args = vars(ap.parse_args())

# boudns of color range in hsv
# ranges are h:[0, 180], s:[0, 255], v:[0, 255]
colorLower = (20, 86, 50)
colorUpper = (40, 255, 255)

# if no video was chosen, use webcam
if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])

frame_num = 0
START_FRAME = 500
FRAME_STRIDE = 3

# loop through frames
while True:
    # get frame from video stream
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    frame_num += 1
    
    # stop if reaches end of video
    if frame is None:
        break
    
    # start at specific frame
    if frame_num < START_FRAME:
        continue
    
    # skip some frames to read faster
    if frame_num % FRAME_STRIDE != 0:
        continue
    
    # resize, blur, and convert from rgb to hsv
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # get mask of areas within color range
    color_mask = cv2.inRange(hsv, colorLower, colorUpper)
    
    # dilate to fill unwanted holes in mask
    #   ^(like the logo or any writing on the balls)
    # over-erode and dilate again to remove random tiny blobs
    color_mask = cv2.dilate(color_mask, None, iterations=5)
    color_mask = cv2.erode(color_mask, None, iterations=10)
    color_mask = cv2.dilate(color_mask, None, iterations=5)
    
    # find local maxima of distance from edges of mask
    D = ndimage.distance_transform_edt(color_mask.copy())
    localMax = peak_local_max(D, indices=False, min_distance=20,
        labels=color_mask)
    
    # watershed transform using local maxima
    # (splits contours that touch/overlap)
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=color_mask)
    
    # loop over labels
    for label in np.unique(labels):
        # background has label 0, so ignore it
        if label == 0:
            continue
        
        new_mask = np.zeros(color_mask.shape, dtype="uint8")
        new_mask[labels == label] = 255
        
        # detect contours in the mask and grab the largest one
        contour_list = cv2.findContours(new_mask.copy(),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_list = imutils.grab_contours(contour_list)
        contour = max(contour_list, key=cv2.contourArea)
        
        # approximate polygon (circle will have a lot of vertices
        # compared to something like a rectangle, so this lets you
        # check if the contour is likely circular)
        hull = cv2.convexHull(contour)
        approx = cv2.approxPolyDP(hull, 3, True)
                
        # bounding circle
        ((x, y), r) = cv2.minEnclosingCircle(hull)
        
        # make sure radius is not tiny and check if seems circular
        if r > 10 and len(approx) > 7:
            # draw a circle enclosing the object
            cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 1)
            cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), 1)
    
    # show frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # stop loop if q key is pressed
    if key == ord("q"):
        break

# stop video stream if no video file is being used
if not args.get("video", False):
    vs.stop()
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()