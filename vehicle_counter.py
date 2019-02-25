import math
import cv2
import numpy as np
from dividing_line import indetector, get_points_dis, get_dividing_line
from pers_warpper import warp_point

# ============================================================================

CAR_COLOURS = [ (0,0,255), (0,106,255), (0,216,255), (0,255,182), (0,255,76)
    , (144,255,0), (255,255,0), (255,148,0), (255,0,178), (220,0,255) ]

# ============================================================================
# class for each vehicle
class Vehicle(object):
    def __init__(self, id, position, direction, M):
        self.id = id    # to identify the vehicle
        self.positions = [position]  # the vehicle's position, most recent at end
        self.warpped_positions = [warp_point(position,M)] # vehicle's position in perspective-warpped frame
        self.speed = [] # vehicle's speed
        self.frames_since_seen = 0  # last seen frame
        self.counted = False    # wether this vehicle be counted
        self.direction = direction

    # func for recording new position and update last seen frame
    def add_position(self, new_position):
        self.positions.append(new_position)
        self.frames_since_seen = 0
    
    # func for warp position point:
    def warp_position(self, M):
        self.warpped_positions.append(warp_point(self.positions[-1],M))

    # func for calculate vehicle speed
    def get_step_speed(self, meters_per_pixle, time_since_last_frame):
        if len(self.warpped_positions) > 1:
            dx = self.warpped_positions[-1][0] - self.warpped_positions[-2][0]
            dy = self.warpped_positions[-1][1] - self.warpped_positions[-2][1]
            step_speed = int(math.sqrt(dx**2+dy**2) * meters_per_pixle * 3.6 / time_since_last_frame)
            self.speed.append(step_speed)

    # func for draw the trajectory
    def draw(self, original_frame, warpped_frame):
        car_colour = CAR_COLOURS[self.id % len(CAR_COLOURS)] # vehicle's id is distinct, so every 10 vehicles will have different colours
        # draw on original frame
        # draw all positions:
        for point in self.positions:
            cv2.circle(original_frame, point, 3, car_colour, -1) # draw position points
        cv2.polylines(original_frame, [np.int32(self.positions)], False, car_colour, 1) # draw polylines between all positions
            # draw id on every trajectory point
            #cv2.putText(output_image, str(self.id), point, cv2.FONT_HERSHEY_PLAIN, 0.7, (127, 255, 255), 1)
        # draw id on vehicle's current position
        #cv2.putText(original_frame, str(self.id), self.positions[-1], cv2.FONT_HERSHEY_PLAIN, 3, (127, 255, 255), 1)
        # draw speed
        cv2.putText(original_frame, '{} KM/H'.format(str(self.speed[-1])), self.positions[-1], cv2.FONT_HERSHEY_PLAIN, 2, (127, 255, 255), 2)
        #=======================================================================================
        # draw on warpped frame:
        for point in self.warpped_positions:
            cv2.circle(warpped_frame, point, 3, car_colour, -1) # draw position points
            cv2.polylines(warpped_frame, [np.int32(self.warpped_positions)], False, car_colour, 1) # draw polylines between all positions
            # draw id on every trajectory point
            #cv2.putText(output_image, str(self.id), point, cv2.FONT_HERSHEY_PLAIN, 0.7, (127, 255, 255), 1)
        # draw id on vehicle's current position
        #cv2.putText(warpped_frame, str(self.id), self.warpped_positions[-1], cv2.FONT_HERSHEY_PLAIN, 3, (127, 255, 255), 1)
        # draw speed
        cv2.putText(warpped_frame, '{} KM/H'.format(str(self.speed[-1])), self.warpped_positions[-1], cv2.FONT_HERSHEY_PLAIN, 2, (127, 255, 255), 2)


# ============================================================================
# class for counting the vehicles
class VehicleCounter(object):
    def __init__(self, shape, detector):
        self.height, self.width = shape # frame size
        self.detector = detector  # virtual loop position ((x1,y1),(x4,y4))
        self.vehicles = []
        self.next_vehicle_id = 0
        self.max_unseen_frames = 7 # when vehicle's frames_since_seen exceed this threshold, remove that vehicle
        self.r_vehicle_count = 0
        self.l1_vehicle_count = 0
        self.l2_vehicle_count = 0
        self.l3_vehicle_count = 0
        self.l4_vehicle_count = 0        

    # if func has no 'self' argument, must @staticmethod, otherwise there will be something wrong with argument number
    @staticmethod
    def update_single_vehicle(vehicle, calculated_points, M, meters_per_pixle, time_since_last_frame):
        for i, new in enumerate(calculated_points):
            x_new, y_new = new.ravel()
            # if find one vehicle's last position equals to old point, add this vehicle's new position
            if get_points_dis(vehicle.positions[-1], (x_new, y_new)) < 50:
                vehicle.add_position((x_new, y_new))
                # warp added position point:
                vehicle.warp_position(M)
                # calculate vehicle speed:
                vehicle.get_step_speed(meters_per_pixle, time_since_last_frame)
                #print('vehicle {} matched and updated\n'.format(vehicle.id))
                return 1 # if found match, end func
        #print('vehicle {} not matched\n'.format(vehicle.id))
        vehicle.frames_since_seen += 1

    def update_all_vehicles(self, calculated_points, M, meters_per_pixle, time_since_last_frame):
        for v in self.vehicles:
            self.update_single_vehicle(v, calculated_points, M, meters_per_pixle, time_since_last_frame)
        # drop long-time unseen vhicles
        self.vehicles = [x for x in self.vehicles if x.frames_since_seen < 11]

    def update_count(self):
        # Count any uncounted vehicles that have past the detector
        for vehicle in self.vehicles:
            # judge direction:
            if vehicle.direction == "right":
                if not vehicle.counted and indetector(self.detector[-1], vehicle.positions[-1]):
                    self.r_vehicle_count += 1
                    vehicle.counted = True
            if vehicle.direction == "left":
                if not vehicle.counted:
                    if indetector(self.detector[0], vehicle.positions[-1]):                    
                        self.l1_vehicle_count += 1
                        vehicle.counted = True
                    elif indetector(self.detector[1], vehicle.positions[-1]):
                        self.l2_vehicle_count += 1
                        vehicle.counted = True
                    elif indetector(self.detector[2], vehicle.positions[-1]):
                        self.l3_vehicle_count += 1
                        vehicle.counted = True
                    elif indetector(self.detector[3], vehicle.positions[-1]):
                        self.l4_vehicle_count += 1
                        vehicle.counted = True
    
    def draw_all_vehicles(self, original_frame, warpped_frame):
        # draw the vehicles on an image
        if (original_frame is not None) and (warpped_frame is not None):
            for vehicle in self.vehicles:
                vehicle.draw(original_frame, warpped_frame)
    
    @staticmethod
    def find_if_a_new_vehicle(obj, calculated_points):
        # check if there are new objects come out:
        if len(calculated_points) == 0:
            return 'yes'
        x, y = obj.ravel() 
        for new_point in calculated_points:
            xn, yn = new_point.ravel()
            dis = get_points_dis((x,y), (xn,yn))
            if dis < 50:
                # this obj is not a new vehicle, return 'no' and func over
                return 'no'
        # no 'new_point' matched this obj is a new vehicle, return 'yes' and func over
        return 'yes'
    
    def creat_a_new_vehicle(self, new_obj, slope, b, M):
        x, y = new_obj.ravel()
        # kill new object in detecting area:
        if indetector(self.detector[0], (x, y)) or indetector(self.detector[1], (x, y)) or indetector(self.detector[2], (x, y)) or indetector(self.detector[3], (x, y)) or indetector(self.detector[4], (x, y)):
            return 0
        # judge vehicle's direction
        if y < x * slope + b:
            direction = 'right'
        else :
            direction = 'left'
        # creat new vehicle object
        new_vehicle = Vehicle(self.next_vehicle_id, (x, y), direction, M) # vehicle ID starts from 1, then increase by 1 for next vehicle
        self.next_vehicle_id += 1
        self.vehicles.append(new_vehicle)
        return 1
# ============================================================================