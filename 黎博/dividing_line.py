import cv2
import numpy as np
import math
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

def indetector(detector, point):
    polygon = Polygon(detector)
    return polygon.contains(Point(point))

def get_dividing_line(p1, p2):
    #dividing line equition:
    k = (p1[1]-p2[1])/(p1[0]-p2[0])
    #b = y1 - x1*k
    b = p1[1] - p1[0]*k
    return k, b

def draw_lane_lines(frame, lane_detector_list, counter):
    # show detectors
    for detector in lane_detector_list:
        cv2.polylines(frame, [np.int32(detector)], True, (0,255,255), 2)
    cv2.putText(frame, "count:{}".format(counter.r1), (lane_detector_list[0][0][0]+20, lane_detector_list[0][0][1]+20), cv2.FONT_HERSHEY_PLAIN, 1.5, (100, 255, 155), 2)
    cv2.putText(frame, "count:{}".format(counter.r2), (lane_detector_list[1][0][0]+20, lane_detector_list[1][0][1]+20), cv2.FONT_HERSHEY_PLAIN, 1.5, (100, 255, 155), 2)
    cv2.putText(frame, "count:{}".format(counter.r3), (lane_detector_list[2][0][0]+20, lane_detector_list[2][0][1]+20), cv2.FONT_HERSHEY_PLAIN, 1.5, (100, 255, 155), 2)
    cv2.putText(frame, "count:{}".format(counter.l1), (lane_detector_list[3][0][0]+20, lane_detector_list[3][0][1]+20), cv2.FONT_HERSHEY_PLAIN, 1.5, (100, 255, 155), 2)


def get_points_dis(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)