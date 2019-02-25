import cv2
import numpy as np
import math
from fg import get_object_to_track
from vehicle_counter import Vehicle, VehicleCounter
from dividing_line import get_dividing_line, draw_lane_lines, get_points_dis
from datetime import datetime

video = "L1.avi"
cap = cv2.VideoCapture(video)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('save.avi',fourcc,30.0,(704,576))
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=True) # filter model detec gery shadows for removing
# filter kernel for denoising:
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# init previous frame:
ret, previous_frame = cap.read()
# convert to gray:
previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03) )                  
# Create some random colors
color = np.random.randint(0,255,(100,3))
frameID = 0
# list for storing new and old points:
calculated_points = []
previous_obj_points = []
#===============================================
slope, b = get_dividing_line((255,0),(651,576))
# init lines
# right rectangle detector:
rd1 = [ (473,375), (519,388), (577,443), (405,424) ]
rd2 = [ (720,265), (846,273), (815,397), (519,388) ]
rd3 = [ (846,273), (1008,277), (1026,417), (815,398)]
# left rectangle detectors:
# detector1:
ld1 = [ (89,388), (405,426), (337, 507), (23, 440) ]
# all detectors, be aware of order:
lane_detectors = [ rd1, rd2, rd3, ld1 ]
#==============================================
# get perspective tansformation matrix
#original_points = np.float32([(0,200), (458,283), (651,576), (0,510)])
#destination_points = np.float32([(0,0), (600,0), (600,400), (0,400)])
#M = cv2.getPerspectiveTransform(original_points, destination_points)
#meters_per_pixle = 4.56/260
#==============================================
count = 0
# vehicle counter obj
counter = None 
#==============================================
now_time = datetime.now()
# main loop:
while True:
    if frameID > 0:
        previous_time = now_time
        now_time = datetime.now()
        time_since_last_frame = (now_time - previous_time).total_seconds()
    ret, frame = cap.read()
    if not ret:
        break
    # get warpped frame
    #warpped_frame = warp_frame_img(frame, M, (600,400))
    # copy original frame
    original_frame = frame.copy()
    # convert to gray for LK flow calculating
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # creat VehicleCounter class object counter at first frame
    if counter is None:
        # We do this here, so that we can initialize with actual frame size
        counter = VehicleCounter(original_frame.shape[:2], lane_detectors)
    # ==========================================================================================
    # only do LK flow when there is points in the last frame
    if len(previous_obj_points) > 0:
        # LK calculated_points and previous_obj_points are co-responding for each vehicle
        calculated_points, st, err = cv2.calcOpticalFlowPyrLK(previous_frame, gray_frame, previous_obj_points, None, **lk_params)
    if len(counter.vehicles) > 0:
        # update all vehicles:
        counter.update_all_vehicles(calculated_points, time_since_last_frame)
        # draw:
        counter.draw_all_vehicles(original_frame)
    # get fg objects
    fg, frame_objects = get_object_to_track(frame, fgbg, kernel)
    # check if there are new vehicles come out:
    for obj in frame_objects:
        flag = counter.find_if_a_new_vehicle(obj, calculated_points)
        if flag == 'yes':
            counter.creat_new_vehicles(obj, slope, b)
    # update count
    counter.update_count()
    # show detectors and counts
    draw_lane_lines(original_frame, lane_detectors, counter)
    # update 'previous frame' and 'previous points' for next frame
    previous_obj_points = frame_objects
    previous_frame = gray_frame
    cv2.imshow('fg', fg)
    cv2.imshow('frame', original_frame)
    #cv2.imshow('warpped', warpped_frame)
    out.write(original_frame)
    frameID += 1
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
