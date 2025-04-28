import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


path = "./kitti_dataset/training/velodyne/000000.bin"

data = np.fromfile(path, dtype=np.float32).reshape(-1, 4)

# 2D visualization
x = data[:, 0]
y = data[:, 1]

plt.figure()
plt.scatter(x, y, s=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.title("2D Bird-eye Visualization")
plt.show()

# 3D visualization
pointcloud = o3d.geometry.PointCloud()
pointcloud.points = o3d.utility.Vector3dVector(data[:, :3])

o3d.visualization.draw_geometries(
    [pointcloud], window_name="3D Visualization", width=800, height=600
)
