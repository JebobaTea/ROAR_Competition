import matplotlib.pyplot as plt
import numpy as np

data = np.load('Monza.npz')
lst = data.files
for item in lst:
    print(item)
    print(data[item])

waypoints = data["locations"]

plt.figure(figsize=(11, 11))
plt.axis((-1100, 1100, -1100, 1100))
plt.tight_layout()

for waypoint in waypoints:
    plt.plot(waypoint[0], waypoint[1], "bo")

plt.show()