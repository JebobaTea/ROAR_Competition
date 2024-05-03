"""
Competition instructions:
Please do not change anything else but fill out the to-do sections.
"""

from typing import List, Tuple, Dict, Optional
import roar_py_interface
import numpy as np
from collections import deque

def normalize_rad(rad : float):
    return (rad + np.pi) % (2 * np.pi) - np.pi

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

    def get_lateral_pid_config(self):
        conf = {
        "60": {
                "Kp": 0.8,
                "Kd": 0.05,
                "Ki": 0.05
        },
        "70": {
                "Kp": 0.7,
                "Kd": 0.07,
                "Ki": 0.07
        },
        "80": {
                "Kp": 0.6,
                "Kd": 0.08,
                "Ki": 0.08
        },
        "90": {
                "Kp": 0.5,
                "Kd": 0.11,
                "Ki": 0.09
        },
        "100": {
                "Kp": 0.45,
                "Kd": 0.12,
                "Ki": 0.1
        },
        "120": {
                "Kp": 0.4,
                "Kd": 0.13,
                "Ki": 0.1
        },
        "130": {
                "Kp": 0.30,
                "Kd": 0.2,
                "Ki": 0.09
        },
        "140": {
                "Kp": 0.25,
                "Kd": 0.2,
                "Ki": 0.09
        },
        "160": {
                "Kp": 0.25,
                "Kd": 0.2,
                "Ki": 0.06
        },
        "180": {
                "Kp": 0.25,
                "Kd": 0.2,
                "Ki": 0.05
        },
        "200": {
                "Kp": 0.28,
                "Kd": 0.2,
                "Ki": 0.04
        },
        "230": {
                "Kp": 0.26,
                "Kd": 0.1,
                "Ki": 0.05
        },
        "300": {
                "Kp": 0.205,
                "Kd": 0.1,
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
         # We use the 3rd waypoint ahead of the current waypoint as the target waypoint
        waypoint_to_follow = self.lat_pid_controller.get_waypoint_at_offset(self.maneuverable_waypoints, self.current_waypoint_idx, 3)
        far_waypoint = self.lat_pid_controller.get_waypoint_at_offset(self.maneuverable_waypoints, self.current_waypoint_idx, 25)
        really_far_waypoint = self.lat_pid_controller.get_waypoint_at_offset(self.maneuverable_waypoints, self.current_waypoint_idx, 40)

        # Calculate delta vector towards the target waypoint
        vector_to_waypoint = (waypoint_to_follow.location - vehicle_location)[:2]
        heading_to_waypoint = np.arctan2(vector_to_waypoint[1],vector_to_waypoint[0])

        # Calculate delta angle towards the target waypoint
        delta_heading = normalize_rad(heading_to_waypoint - vehicle_rotation[2])

        # Proportional controller to steer the vehicle towards the target waypoint
        steer_control = self.lat_pid_controller.run_in_series(vehicle_location, vehicle_rotation, speed, waypoint_to_follow)
        far_error = self.lat_pid_controller.find_waypoint_error(vehicle_location, vehicle_rotation, speed, far_waypoint)
        really_far_error = self.lat_pid_controller.find_waypoint_error(vehicle_location, vehicle_rotation, speed, really_far_waypoint)

        throttle = 1
        brake = 0
        if speed > 200:
           throttle = 0.7
        if abs(steer_control) > 0.3 and speed > 40:
            throttle = 0
            brake = 1
        elif abs(steer_control) > 0.05 and speed > 120:
            throttle = 0
            brake = 1
        elif abs(far_error) > 0.5 and speed > 60:
            throttle = 0
            brake = 1
        elif abs(far_error) > 0.1 and speed > 80:
            throttle = 0
            brake = 1
        elif abs(far_error) > 0.05 and speed > 100:
            throttle = 0
            brake = 1
        elif abs(really_far_error) > 0.1 and speed > 140:
            throttle = 0
            brake = 1
        elif abs(really_far_error) > 0.05 and speed > 120:
            throttle = 0
            brake = 1

        print(round(far_error * 100)/100)
        print(round(really_far_error * 100)/100)
        print(round(speed))

        control = {
            "throttle": np.clip(throttle, 0.0, 1.0),
            "steer": steer_control,
            "brake": np.clip(brake, 0.0, 1.0),
            "hand_brake": 0.0,
            "reverse": 0,
            "target_gear": 0
        }
        await self.vehicle.apply_action(control)
        return control

class LatPIDController():
    def __init__(self, config: dict, dt: float = 0.05):
        self.config = config
        self.steering_boundary = (-1.0, 1.0)
        self._error_buffer = deque(maxlen=10)
        self._dt = dt

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

    def find_k_values(self, current_speed: float, config: dict) -> np.array:
        k_p, k_d, k_i = 1, 0, 0
        for speed_upper_bound, kvalues in config.items():
            speed_upper_bound = float(speed_upper_bound)
            if current_speed < speed_upper_bound:
                k_p, k_d, k_i = kvalues["Kp"], kvalues["Kd"], kvalues["Ki"]
                break
        return np.array([k_p, k_d, k_i])

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

    def get_waypoint_at_offset(self, maneuverable_waypoints, current_index, offset):
        return maneuverable_waypoints[(current_index + offset) % len(maneuverable_waypoints)]
