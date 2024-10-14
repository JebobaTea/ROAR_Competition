import roar_py_carla
import roar_py_interface
import carla
import numpy as np
from typing import List, Tuple
import transforms3d as tr3d

carla_client = carla.Client('localhost', 2000)
carla_client.set_timeout(15.0)
roar_py_instance = roar_py_carla.RoarPyCarlaInstance(carla_client)
roar_py_world = roar_py_instance.world
roar_py_world.set_asynchronous(True)
roar_py_world.set_control_steps(0.00, 0.005)

waypoint_list : List[roar_py_interface.RoarPyWaypoint] = []
trace_waypoint_laneid = 1
print("LaneIDS: ", roar_py_world.comprehensive_waypoints.keys())
for lane_id in [0, 1]:
    lane_waypoints = roar_py_world.comprehensive_waypoints[lane_id]
    assert len(lane_waypoints) % 2 == 0
    for i in range(len(lane_waypoints)//2):
        first_waypoint = lane_waypoints[2*i]
        second_waypoint = lane_waypoints[2*i+1]
        real_waypoint = roar_py_interface.RoarPyWaypoint.from_line_representation(
            first_waypoint.location,
            second_waypoint.location,
            first_waypoint.roll_pitch_yaw
        )
        # print(real_waypoint)
        waypoint_list.append(real_waypoint)


roar_py_instance.close()
np.savez_compressed("Monza.npz", **roar_py_interface.RoarPyWaypoint.save_waypoint_list(waypoint_list))

for waypoint in waypoint_list:
    origin_loc = waypoint.location
    forward_loc = waypoint.location + tr3d.euler.euler2mat(*waypoint.roll_pitch_yaw) @ np.array([0.5,0.0,0.0])
    roar_py_world.carla_world.debug.draw_arrow(
        roar_py_carla.location_to_carla(origin_loc),
        roar_py_carla.location_to_carla(forward_loc),
        thickness=0.1,
        arrow_size=0.1,
        color=carla.Color(0,255,0),
        life_time=-1.0
    )