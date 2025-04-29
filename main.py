import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


def read_label(path: str):
    boxes = []

    with open(path, "r") as f:
        for line in f.readlines():
            object = line.strip().split(" ")

            name = object[0]
            height = float(object[8])
            width = float(object[9])
            length = float(object[10])
            x = float(object[11])
            y = float(object[12])
            z = float(object[13])
            yaw = float(object[14])

            box = {
                "name": name,
                "height": height,
                "width": width,
                "length": length,
                "x": x,
                "y": y,
                "z": z,
                "yaw": yaw,
            }
            boxes.append(box)

    return boxes


def read_calib(path: str):
    calibs = {}
    with open(path, "r") as f:
        for line in f.readlines():
            values = line.strip().split(": ")
            if len(values) < 2:
                continue
            key = values[0]
            values = values[1]
            calibs[key] = np.array([float(x) for x in values.strip().split()])

    calibs["R0_rect"] = calibs["R0_rect"].reshape(3, 3)
    calibs["Tr_velo_to_cam"] = calibs["Tr_velo_to_cam"].reshape(3, 4)
    calibs["Tr_cam_to_velo"] = np.vstack([calibs["Tr_velo_to_cam"], [0, 0, 0, 1]])
    calibs["Tr_cam_to_velo"] = np.linalg.inv(calibs["Tr_cam_to_velo"])

    return calibs


def draw_3d_boxes(pointcloud, boxes, calib):
    all_lines = []
    all_points = []

    for i, box in enumerate(boxes):
        h = box["height"]
        w = box["width"]
        l = box["length"]

        x = box["x"]
        y = box["y"]
        z = box["z"]

        yaw = box["yaw"]

        x_corners = np.array(
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        )
        y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
        z_corners = np.array(
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        )
        corners = np.vstack([x_corners, y_corners, z_corners])

        R = np.array(
            [
                [np.cos(yaw), 0, np.sin(yaw)],
                [0, 1, 0],
                [-np.sin(yaw), 0, np.cos(yaw)],
            ]
        )

        corners = np.dot(R, corners)
        corners += np.vstack([x, y, z])

        Tr = calib["Tr_cam_to_velo"]
        box_cam = np.vstack([corners, np.ones((1, 8))])
        box_velo = Tr @ box_cam
        all_points.append(box_velo[:3].T)

        lines = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]

        base_index = i * 8
        for line in lines:
            all_lines.append([line[0] + base_index, line[1] + base_index])

    all_points = np.vstack(all_points)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_points)
    line_set.lines = o3d.utility.Vector2iVector(all_lines)

    o3d.visualization.draw_geometries(
        [pointcloud, line_set],
        window_name="3D Point Cloud Visualization with Bounding Boxes",
        width=800,
        height=600,
    )


if __name__ == "__main__":
    data_path = "./kitti_dataset/training/pointclouds/000000.bin"
    label_path = "./kitti_dataset/training/labels/000000.txt"
    calib_path = "./kitti_dataset/training/calib/000000.txt"

    boxes = read_label(label_path)
    calib = read_calib(calib_path)
    data = np.fromfile(data_path, dtype=np.float32).reshape(-1, 4)[:, :3]

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(data[:, :3])

    draw_3d_boxes(pointcloud, boxes, calib)

    # # 2D visualization
    # x = data[:, 0]
    # y = data[:, 1]

    # plt.figure()
    # plt.scatter(x, y, s=0.5)
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.title("2D Bird-eye Visualization")
    # plt.show()
