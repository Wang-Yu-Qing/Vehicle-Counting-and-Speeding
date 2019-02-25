import cv2
import numpy as np

def get_object_to_track(frame, fgbg, kernel): 
    #====================== get and filter foreground mask ================
    #print(original_frame.shape[:2])
    fgmask = fgbg.apply(frame)
    #==================================================================
    # Fill any small holes
    closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    # Remove noise
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    # Dilate to merge adjacent blobs
    dilation = cv2.dilate(opening, kernel, iterations = 2)
    # threshold (remove grey shadows)
    dilation[dilation < 240] = 0
    #=====================================================
    # min object size
    MIN_CONTOUR_WIDTH = 50
    MIN_CONTOUR_HEIGHT = 50
    # Find the contours of any vehicles in the frame(dilation)
    im, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # list for taking qualified centroids
    matches = []
    # for every contour getted, get their centroids
    for contour in contours:
        # kill small contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        contour_valid = (w >= MIN_CONTOUR_WIDTH) and (h >= MIN_CONTOUR_HEIGHT)
        # if too small, go to next loop step
        if not contour_valid:
            continue
        # get qualified contour's centroid
        centroid = [[x + w/2, y + h/2]]
        matches.append(centroid)
    points_to_track = np.asarray(matches, dtype=np.float32)
    return dilation, points_to_track


