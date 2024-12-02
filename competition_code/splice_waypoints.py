import roar_py_interface
from typing import List
import numpy as np

def distanceToWaypoint(currentLoc, waypoint: roar_py_interface.RoarPyWaypoint):
    return np.linalg.norm(currentLoc[:2] - waypoint.location[:2])

def findCurrentIndex(currentLoc, waypoints: List[roar_py_interface.RoarPyWaypoint]):
    for i in range(0, len(waypoints)):
        if distanceToWaypoint(currentLoc, waypoints[i % len(waypoints)]) < 4:
            return i % len(waypoints)
    return 0

base = roar_py_interface.RoarPyWaypoint.load_waypoint_list(
    np.load("waypoints9.npz")
)

modified = roar_py_interface.RoarPyWaypoint.load_waypoint_list(
    np.load("waypoints.npz")
)

start = findCurrentIndex(modified[0].location, base)
print(start)
new = []
for i in range(0, start):
    new.append(base[i])
for i in range(len(modified)):
    new.append(modified[i])
for i in range(start + len(modified), len(base)):
    new.append(base[i])

np.savez_compressed(
    "test.npz",
    **roar_py_interface.RoarPyWaypoint.save_waypoint_list(new),
)
