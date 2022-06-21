from scipy.spatial.transform import Rotation
import numpy as np
import plotly.graph_objs as go
import pers


def plot_skeleton(filename):
    points = np.loadtxt(f"{filename}.csv", delimiter=",")
    skeleton_lines = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [0, 11], [0, 12],
                      [11, 13], [12, 14], [13, 15], [14, 16]]

    lay = go.Layout(autosize=True, scene=dict(camera=dict(eye=dict(x=1.15, y=1.15, z=0.8)), aspectmode='data'))
    fig = go.Figure(layout=lay)
    fig.add_trace(
        go.Scatter3d(x=points[:, 0].flatten(), y=points[:, 1].flatten(), z=points[:, 2].flatten(), mode='markers'))

    for line in skeleton_lines:
        fig.add_trace(
            go.Scatter3d(x=points[line, 0].flatten(), y=points[line, 1].flatten(), z=points[line, 2].flatten(),
                         mode='lines'))

    fig.show()


if __name__ == "__main__":
    folder = "human5-exp2"

    cam_matrix = np.array([[120 / np.tan(45 / 2), 0, 120], [0, 80 / np.tan(45 / 2), 80], [0, 0, 1]])
    ceiling = [(0.5, 0., 2.99), Rotation.from_euler('XYZ', [180, 0, 0], degrees=True)]
    ground = [(0.5, 0., 0.02), Rotation.from_euler('XYZ', [0, 0, 0], degrees=True)]
    wall1 = [(2.99, 0, 1.5), Rotation.from_euler('XYZ', [0, -90, 0], degrees=True)]
    wall2 = [(-2.98, 0, 1.5), Rotation.from_euler('XYZ', [0, 90, 0], degrees=True)]
    wall3 = [(0, -1.98, 1.5), Rotation.from_euler('XYZ', [-90, 0, 0], degrees=True)]
    wall4 = [(0, 1.99, 1.5), Rotation.from_euler('XYZ', [90, 0, 0], degrees=True)]
    sensors = np.array([ceiling, ground, wall1, wall2, wall3, wall4], dtype=object)

    boolean_mask_lidar = np.array([1, 1, 1, 1, 1, 1])
    boolean_mask_rgbd = np.array([0, 0, 0, 0, 0, 0])

    lidar_poses = sensors[boolean_mask_lidar == 1, 0]
    rgbd_poses = sensors[boolean_mask_rgbd == 1, 0]
    cam_rotations = sensors[boolean_mask_rgbd == 1, 1]
    cam_matrices = []
    for i in range(np.sum(boolean_mask_rgbd)):
        cam_matrices.append(cam_matrix)
    
    resolution = 0.05
    output_name = "pokus"

    # PADS poses - ((x_min, x_max), (y_min, y_max))
    pad_poses = []  # [((-0.2, 1), (-1, -0.6)), ((-0.2, 1), (1, 1.5))]
    # GATES poses - ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    gate_poses = []  # [((-2.8, 2.8), (-0.7, -0.7), (1, 1))]

    proximity_poses = []
    proximity_rays = []
    robot_inflation_value = 0.3  # robot proximity inflation in meters
    proximity_range = 10  # max distance detected by ray proximity sensor
    lidar_range = 10  # max distance detected by lidar (fish-eye) sensor
    simulation = pers.Pers(folder, resolution=resolution, output_name=output_name, lidar_poses=lidar_poses,
                           rgbd_poses=rgbd_poses, pad_poses=pad_poses, gate_poses=gate_poses,
                           proximity_poses=proximity_poses, cam_matrices=cam_matrices, cam_rotations=cam_rotations,
                           proximity_rays=proximity_rays, robot_inflation_value=robot_inflation_value,
                           proximity_range=proximity_range, lidar_range=lidar_range)

    simulation.process_sensors(compute_statistics=True)

    # plot_skeleton(f"results/{output_name}/keypoints0")
