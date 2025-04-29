import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def read_label(path: str):
    boxes = []

    with open(path, "r") as f:
        objects = f.readlines()
        for object in objects:
            object = object.strip().split(" ")

            name = object[0]
            height = float(object[8])
            width = float(object[9])
            length = float(object[10])
            x = float(object[11])
            y = float(object[12])
            z = float(object[13])
            rotation = float(object[14])

            box = {
                "name": name,
                "height": height,
                "width": width,
                "length": length,
                "x": x,
                "y": y,
                "z": z,
                "rotation": rotation,
            }
            boxes.append(box)

    return boxes


if __name__ == "__main__":
    data_path = "./kitti_dataset/training/pointclouds/000000.bin"
    label_path = "./kitti_dataset/training/labels/000000.txt"

    boxes = read_label(label_path)
    print(boxes)

    # data = np.fromfile(data_path, dtype=np.float32).reshape(-1, 4)

    # # 2D visualization
    # x = data[:, 0]
    # y = data[:, 1]

    # plt.figure()
    # plt.scatter(x, y, s=0.5)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("2D Bird-eye Visualization")
    # plt.show()

    # # 3D visualization
    # pointcloud = o3d.geometry.PointCloud()
    # pointcloud.points = o3d.utility.Vector3dVector(data[:, :3])

    # o3d.visualization.draw_geometries(
    #     [pointcloud], window_name="3D Visualization", width=800, height=600
    # )
