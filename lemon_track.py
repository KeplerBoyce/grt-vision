# imports
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import math

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
args = vars(ap.parse_args())

colorLower = (20, 86, 6)
colorUpper = (40, 255, 255)

if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])

while True:
    # get frame from video stream
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    
    # stop if reaches end of video
    if frame is None:
        break
    
    # resize, blur, and convert from rgb to hsv
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # mask for color
    # erosions and dilations remove small blobs in mask
    mask = cv2.inRange(hsv, colorLower, colorUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    contour_list = cv2.findContours(mask.copy(), cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)
    contour_list = imutils.grab_contours(contour_list)
    
    # check that at least 1 contour was found
    if contour_list is not None:
    
        # loop through all contours
        for contour in contour_list:
        
            # find enclosing circle and center of largest contour
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            M = cv2.moments(contour)
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
            
            # check that contour is not tiny
            if radius > 10 and len(approx) > 7:
            
                # draw circle and center on frame
                cv2.circle(frame, (int(x), int(y)), int(radius),
                    (255, 0, 255), 1)
                cv2.circle(frame, center, 5, (255, 255, 0), -1)
    
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