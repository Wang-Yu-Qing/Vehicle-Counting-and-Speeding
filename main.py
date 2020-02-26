import cv2
import numpy as np
import math
from foreground import get_object_to_track
from vehicle_counter import Vehicle, VehicleCounter
from dividing_line import get_dividing_line, draw_lane_lines, get_points_dis
from pers_warpper import warp_frame_img
from datetime import datetime

VIDEO = "saved.avi"

cap = cv2.VideoCapture(VIDEO)

# # for video writing
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('save.avi',fourcc,30.0,(704,576))

# init lines
# right rectangle detector:
RIGHT_DETECTOR = [ (255,0), (410,0), (579,192), (362,192) ]
# left rectangle detectors:
# detector1:
LD1 = [ (12,293), (149,314), (190, 576), (5, 576) ]
# detector2:
LD2 = [ (149,314), (273,360), (346,576), (190,576) ]
# detector3:
LD3 = [ (273,360), (376,376), (469,576), (346,576) ]
# detector4:
LD4 = [ (376,376), (486,387), (611,576), (469,576) ]
# all detectors, be aware of order:
LANE_DETECTORS = [ LD1, LD2, LD3, LD4, RIGHT_DETECTOR ]

# get perspective tansformation matrix
ORIGINAL_POINTS = np.float32([(0,200), (458,283), (651,576), (0,510)])
DESTINATION_POINTS = np.float32([(0,0), (600,0), (600,400), (0,400)])
M = cv2.getPerspectiveTransform(ORIGINAL_POINTS, DESTINATION_POINTS)
M_PER_PIXLE = 4.56/260


def main_loop():
    # foreground utils
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=True) # filter model detec gery shadows for removing
    # filter kernel for denoising:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    # init previous frame:
    ret, previous_frame = cap.read()
    # convert to gray:
    previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03) )                  
    frameID = 0
    # list for storing new and old points:
    calculated_points = []
    previous_obj_points = []
    slope, b = get_dividing_line((255,0),(651,576))
    count = 0
    # vehicle counter obj
    counter = None 
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
        warpped_frame = warp_frame_img(frame, M, (600,400))
        # copy original frame
        original_frame = frame.copy()
        # convert to gray for LK flow calculating
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # creat VehicleCounter class object counter at first frame
        if counter is None:
            # We do this here, so that we can initialize with actual frame size
            counter = VehicleCounter(original_frame.shape[:2], LANE_DETECTORS)
        # ==========================================================================================
        # only do LK flow when there is points in the last frame
        if len(previous_obj_points) > 0:
            # LK calculated_points and previous_obj_points are co-responding for each vehicle
            calculated_points, st, err = cv2.calcOpticalFlowPyrLK(previous_frame, gray_frame, previous_obj_points, None, **lk_params)
        if len(counter.vehicles) > 0:
            # update all vehicles:
            counter.update_all_vehicles(calculated_points, M, M_PER_PIXLE, time_since_last_frame)
            # draw:
            counter.draw_all_vehicles(original_frame, warpped_frame)
        # get fg objects
        fg, frame_objects = get_object_to_track(frame, fgbg, kernel)
        # check if there are new vehicles come out:
        if len(frame_objects) > 0:
            for obj in frame_objects:
                flag = counter.find_if_a_new_vehicle(obj, calculated_points)
                if flag == 'yes':
                    counter.creat_a_new_vehicle(obj, slope, b, M)
        # update count
        counter.update_count()
        # show detectors and counts
        draw_lane_lines(original_frame, LANE_DETECTORS, counter)
        # update 'previous frame' and 'previous points' for next frame
        previous_obj_points = frame_objects
        previous_frame = gray_frame
        cv2.imshow('fg', fg)
        cv2.imshow('frame', original_frame)
        cv2.imshow('warpped', warpped_frame)
        # out.write(original_frame)
        frameID += 1
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main_loop()