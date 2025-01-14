import roar_py_interface
import numpy as np
import os
from collections import deque
from typing import List, Tuple, Dict, Optional

def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

# Util function to find closest waypoint
def filter_waypoints(location : np.ndarray, current_idx: int, waypoints : List[roar_py_interface.RoarPyWaypoint]) -> int:
    def dist_to_waypoint(waypoint : roar_py_interface.RoarPyWaypoint):
        return np.linalg.norm(
            location[:2] - waypoint.location[:2]
        )
    for i in range(current_idx, len(waypoints) + current_idx):
        if dist_to_waypoint(waypoints[i%len(waypoints)]) < 3:
            return i % len(waypoints)
    return current_idx

class RoarCompetitionSolution:
    def __init__(
        self,
        maneuverable_waypoints: List[roar_py_interface.RoarPyWaypoint],
        vehicle : roar_py_interface.RoarPyActor,
        camera_sensor : roar_py_interface.RoarPyCameraSensor = None,
        location_sensor : roar_py_interface.RoarPyLocationInWorldSensor = None,
        velocity_sensor : roar_py_interface.RoarPyVelocimeterSensor = None,
        rpy_sensor : roar_py_interface.RoarPyRollPitchYawSensor = None,
        occupancy_map_sensor : roar_py_interface.RoarPyOccupancyMapSensor = None,
        collision_sensor : roar_py_interface.RoarPyCollisionSensor = None,
    ) -> None:
        self.maneuverable_waypoints = maneuverable_waypoints
        self.vehicle = vehicle
        self.camera_sensor = camera_sensor
        self.location_sensor = location_sensor
        self.velocity_sensor = velocity_sensor
        self.rpy_sensor = rpy_sensor
        self.occupancy_map_sensor = occupancy_map_sensor
        self.collision_sensor = collision_sensor
        self.lat_pid_controller = LatPIDController(config=self.get_lateral_pid_config())

    # Modify PID equation coefficients depending on speed
    # Refer to chart on the slides for the effects of raising/lowering each individual parameter
    def get_lateral_pid_config(self):
        conf = {
        "60": {
                "Kp": 0.7,
                "Kd": 0.05,
                "Ki": 0.05
        },
        "70": {
                "Kp": 0.7,
                "Kd": 0.07,
                "Ki": 0.07
        },
        "80": {
                "Kp": 0.65,
                "Kd": 0.08,
                "Ki": 0.08
        },
        "90": {
                "Kp": 0.57,
                "Kd": 0.13,
                "Ki": 0.09
        },
        "100": {
                "Kp": 0.5,
                "Kd": 0.15,
                "Ki": 0.1
        },
        "120": {
                "Kp": 0.4,
                "Kd": 0.15,
                "Ki": 0.1
        },
        "130": {
                "Kp": 0.35,
                "Kd": 0.15,
                "Ki": 0.09
        },
        "140": {
                "Kp": 0.3,
                "Kd": 0.15,
                "Ki": 0.09
        },
        "160": {
                "Kp": 0.25,
                "Kd": 0.3,
                "Ki": 0.06
        },
        "180": {
                "Kp": 0.15,
                "Kd": 0.25,
                "Ki": 0.05
        },
        "200": {
                "Kp": 0.01,
                "Kd": 0.4,
                "Ki": 0.04
        },
        "230": {
                "Kp": 0.01,
                "Kd": 0.4,
                "Ki": 0.05
        },
        "300": {
                "Kp": 0.02,
                "Kd": 0.3,
                "Ki": 0.017
        }
        }
        return conf


    async def initialize(self) -> None:
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()

        self.current_waypoint_idx = 10
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )

    # Called continuously during simulation (basically like a gameloop)
    async def step(
        self
    ) -> None:
        # Receive location, rotation and velocity data
        vehicle_location = self.location_sensor.get_last_gym_observation()
        vehicle_rotation = self.rpy_sensor.get_last_gym_observation()
        vehicle_velocity = self.velocity_sensor.get_last_gym_observation()
        vehicle_velocity_norm = np.linalg.norm(vehicle_velocity)
        speed = vehicle_velocity_norm * 3.6


        # Find the waypoint closest to the vehicle
        self.current_waypoint_idx = filter_waypoints(
            vehicle_location,
            self.current_waypoint_idx,
            self.maneuverable_waypoints
        )
         # Steering control is always determined by the waypoint 3 ahead of the current waypoint
        waypoint_to_follow = self.lat_pid_controller.get_waypoint_at_offset(self.maneuverable_waypoints, self.current_waypoint_idx, 3)

        # To determine when we slow down in anticipation of a turn, we select a waypoint ___ indices ahead of the current waypoint depending on the speed
        # The lookahead variable refers to the waypoint in question
        # To get a reference to a waypoint 20 waypoints ahead of the current waypoint, for example, use 
        #   waypoint_20_ahead = self.lat_pid_controller.get_waypoint_at_offset(self.maneuverable_waypoints, self.current_waypoint_idx, 20)
        # Last competition round's winning submission actually used a predefined configuration dictionary instead of scaling offsets as a function of speed, see below:
        # speed_to_lookahead_dict = {
        #    90: 8,
        #    110: 12,
        #    130: 14,
        #    160: 18,
        #    180: 22,
        #    200: 26,
        #    250: 30,
        #    300: 35,
        # }
        if speed > 160:
            lookahead = self.lat_pid_controller.get_waypoint_at_offset(self.maneuverable_waypoints, self.current_waypoint_idx, int(speed/2.5))
        elif speed > 100:
            lookahead = self.lat_pid_controller.get_waypoint_at_offset(self.maneuverable_waypoints, self.current_waypoint_idx, int(speed/3))
        elif speed > 80:
            lookahead = self.lat_pid_controller.get_waypoint_at_offset(self.maneuverable_waypoints, self.current_waypoint_idx, int(speed/10))
        else:
            lookahead = self.lat_pid_controller.get_waypoint_at_offset(self.maneuverable_waypoints, self.current_waypoint_idx, int(speed/15))

        # Calculate delta vector towards the target waypoint
        vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1],vector_to_waypoint[0])

        # Calculate delta angle towards the target waypoint (basically, where do we need to steer towards?)
        delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])

        # Proportional controller to steer the vehicle towards the target waypoint 
        # Handles steering for us - we don't need to worry about this
        steer_control = self.lat_pid_controller.run_in_series(vehicle_location, vehicle_rotation, speed, waypoint_to_follow)

        # Gives us an error value telling us how much we have deviated from the intended path of travel
        # If you want to find the error specifically for a location 100 waypoints ahead, you can use this code:
        #  really_far_waypoint = self.lat_pid_controller.get_waypoint_at_offset(self.maneuverable_waypoints, self.current_waypoint_idx, 100)
        #  error = self.lat_pid_controller.find_waypoint_error(vehicle_location, vehicle_rotation, speed, really_far_waypoint)
        # As our selected waypoint's offset amount is a function of our speed, the error calculation will "look further ahead" at higher speeds
        error = self.lat_pid_controller.find_waypoint_error(vehicle_location, vehicle_rotation, speed, lookahead)

        # Throttle and brake are values from 0 to 1, any values beyond this range will be clipped
        # Intuitively, the higher brake is, the faster you slow down, the higher throttle is, the faster your speed up
        # By default, our vehicle has the pedal to the floor
        throttle = 1
        brake = 0
        os.system("cls")

        # Error can be either negative or positive, so that's why we take the absolute value
        # It's best to observe recordings in slow motion to review exactly when these conditions are met, but the general intentions are:
        # (1) Regardless of speed, when error grows too large, emergency brake
        # (2) When both speed and error are high, slow down
        # (3) When speed is high but error is moderate, decelerate slightly in order to avoid fishtailing
        # (4) At high speeds, stop acceleration but maintain velocity to avoid burning precious momentum while preventing fishtailing
        #     - when PID starts panicking at high speeds, it will start to swing further and further out of control even if you config the coefficients to be less aggressive
        #     - this can partially be fixed by correcting hiccups and weird offsets in the waypoints
        if abs(error) > 0.3 and speed > 80:
            print("Control Case 1")
            throttle = 0.1
            brake = 0.9
            if (speed > 140):
                print("Control Case 1a")
                throttle = 0
                brake = 1
        elif abs(error) > 0.2 and speed > 120:
            print("Control Case 2")
            throttle = 0.5
            brake = 0.5
        elif abs(error) > 0.1 and speed > 150:
            print("Control Case 3")
            throttle = 0.8
            brake = 0.2
        elif abs(error) > 0.05 and speed > 160:
            print("Control Case 4")
            throttle = 0.8
            brake = 0

        # Debug information, feel free to add more as you test
        print("Steer: " + str(round(steer_control * 100)/100))
        print("Error: " + str(round(error * 100)/100))
        print("Speed: " + str(round(speed)))
        print("K-Values: " + str(self.lat_pid_controller.find_k_values(speed, self.get_lateral_pid_config())))
        print("Throttle: " + str(throttle))
        print("Brake: " + str(brake))
        print("Target Gear: " + str(max(1, int(speed / 40))))

        # Don't worry about this - though we definitely should implement gear shifting functionality
        # For reference, last comp's winning solution scaled gear based on speed: gear = max(1, (int)(current_speed**1.15 / 96))
        control = {
            "throttle": np.clip(throttle, 0.0, 1.0),
            "steer": steer_control,
            "brake": np.clip(brake, 0.0, 1.0),
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": max(1, int(speed / 60))
        }
        await self.vehicle.apply_action(control)
        return control

class LatPIDController():
    def __init__(self, config: dict, dt: float = 0.05):
        self.config = config
        self.steering_boundary = (-1.0, 1.0)
        self._error_buffer = deque(maxlen=10)
        self._dt = dt

    # PID equation implementation, decides where to steer
    def run_in_series(self, vehicle_location, vehicle_rotation, current_speed, next_waypoint) -> float:
        v_begin = vehicle_location
        direction_vector = np.array([
            np.cos(normalize_rad(vehicle_rotation[2])),
            np.sin(normalize_rad(vehicle_rotation[2])),
            0])
        v_end = v_begin + direction_vector

        v_vec = np.array([(v_end[0] - v_begin[0]), (v_end[1] - v_begin[1]), 0])

        w_vec = np.array(
            [
                next_waypoint.location[0] - v_begin[0],
                next_waypoint.location[1] - v_begin[1],
                0,
            ]
        )

        v_vec_normed = v_vec / np.linalg.norm(v_vec)
        w_vec_normed = w_vec / np.linalg.norm(w_vec)
        error = np.arccos(min(max(v_vec_normed @ w_vec_normed.T, -1), 1))
        _cross = np.cross(v_vec_normed, w_vec_normed)

        if _cross[2] > 0:
            error *= -1
        self._error_buffer.append(error)
        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        k_p, k_d, k_i = self.find_k_values(current_speed=current_speed, config=self.config)

        lat_control = float(
            np.clip((k_p * error) + (k_d * _de) + (k_i * _ie), self.steering_boundary[0], self.steering_boundary[1])
        )

        return lat_control

    # Feeds the coefficients we specify in the dictionary above into a format the program can understand better
    def find_k_values(self, current_speed: float, config: dict) -> np.array:
        k_p, k_d, k_i = 1, 0, 0
        for speed_upper_bound, kvalues in config.items():
            speed_upper_bound = float(speed_upper_bound)
            if current_speed < speed_upper_bound:
                k_p, k_d, k_i = kvalues["Kp"], kvalues["Kd"], kvalues["Ki"]
                break
        return np.array([k_p, k_d, k_i])

    # A cut-down and simplified version of run_in_series, but only yields an error value instead
    def find_waypoint_error(self, vehicle_location, vehicle_rotation, current_speed, waypoint) -> float:
        v_begin = vehicle_location
        direction_vector = np.array([
            np.cos(normalize_rad(vehicle_rotation[2])),
            np.sin(normalize_rad(vehicle_rotation[2])),
            0])
        v_end = v_begin + direction_vector

        v_vec = np.array([(v_end[0] - v_begin[0]), (v_end[1] - v_begin[1]), 0])

        w_vec = np.array(
            [
                waypoint.location[0] - v_begin[0],
                waypoint.location[1] - v_begin[1],
                0,
            ]
        )

        v_vec_normed = v_vec / np.linalg.norm(v_vec)
        w_vec_normed = w_vec / np.linalg.norm(w_vec)
        error = np.arccos(min(max(v_vec_normed @ w_vec_normed.T, -1), 1))

        return error

    # Get the waypoint ___ indices ahead of the current (use offset parameter)
    def get_waypoint_at_offset(self, maneuverable_waypoints, current_index, offset):
        return maneuverable_waypoints[(current_index + offset) % len(maneuverable_waypoints)]
