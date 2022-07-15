import json
import numpy as np
import plotly.graph_objs as go

import pers

from scipy.spatial.transform import Rotation
from datetime import datetime


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


def generate_experiments(exp_cfg, cases, res, time_stamping=True):

    time_stamp = ''
    if time_stamping:
        time_stamp = datetime.now().strftime("%y%m%d%H%M")

    cam_matrix = np.array([[120 / np.tan(45 / 2), 0, 120], [0, 80 / np.tan(45 / 2), 80], [0, 0, 1]])
    ceiling = [(0.5, 0., 2.99), Rotation.from_euler('XYZ', [180, 0, 0], degrees=True)]
    ground = [(0.5, 0., res+0.01), Rotation.from_euler('XYZ', [0, 0, 0], degrees=True)]
    wall1 = [(2.99, 0, 1.5), Rotation.from_euler('XYZ', [0, -90, 0], degrees=True)]
    wall2 = [(-2.98, 0, 1.5), Rotation.from_euler('XYZ', [0, 90, 0], degrees=True)]
    wall3 = [(0, -1.98, 1.5), Rotation.from_euler('XYZ', [-90, 0, 0], degrees=True)]
    wall4 = [(0, 1.99, 1.5), Rotation.from_euler('XYZ', [90, 0, 0], degrees=True)]
    sensors = np.array([ceiling, ground, wall1, wall2, wall3, wall4], dtype=object)

    experiments = {}
    for exp_key in exp_cfg:
        for case in cases:
            exp = exp_cfg[exp_key]
            code = exp['code'] + '_' + case

            lidar_poses = sensors[np.array(exp['boolean_mask_lidar']) == 1, 0]
            rgbd_poses = sensors[np.array(exp['boolean_mask_rgbd']) == 1, 0]
            cam_rotations = sensors[np.array(exp['boolean_mask_rgbd']) == 1, 1]
            cam_matrices = []
            for i in range(np.sum(exp['boolean_mask_rgbd'])):
                cam_matrices.append(cam_matrix)

            experiments[code] = {'case': case,
                                 'name': time_stamp + exp['name'],
                                 'lidar_poses': lidar_poses,
                                 'rgbd_poses': rgbd_poses,
                                 'pad_poses': list(exp['pad_poses']),
                                 'gate_poses': list(exp['gate_poses']),
                                 'proximity_poses': exp['proximity_poses'],
                                 'proximity_rays': exp['proximity_rays'],
                                 'robot_inflation_value': exp['robot_inflation_value'],
                                 'cam_matrices': cam_matrices,
                                 'cam_rotations': cam_rotations,
                                 }
    return experiments


if __name__ == "__main__":
    cases = ["human1-exp2",
             # "human4-exp2",
             # "human6-exp2",
             # "human63-exp2",
             ]

    with open('experiments.json', "r") as cfg:
        experiment_config = json.load(cfg)

    resolution = 0.02  # m
    experiments = generate_experiments(experiment_config, cases, resolution, time_stamping=False)

    # General parameters
    proximity_range = 10  # max distance detected by ray proximity sensor
    lidar_range = 10  # max distance detected by lidar (fish-eye) sensor
    for exp_key in experiments:
        exp = experiments[exp_key]
        simulation = pers.Pers(exp['case'], resolution=resolution, output_name=exp['name'],
                               lidar_poses=exp['lidar_poses'], rgbd_poses=exp['rgbd_poses'],
                               pad_poses=exp['pad_poses'], gate_poses=exp['gate_poses'],
                               proximity_poses=exp['proximity_poses'], cam_matrices=exp['cam_matrices'],
                               cam_rotations=exp['cam_rotations'],
                               proximity_rays=exp['proximity_rays'],
                               robot_inflation_value=exp['robot_inflation_value'],
                               proximity_range=proximity_range, lidar_range=lidar_range)

        simulation.process_sensors(compute_statistics=True, detect_keypoints=True, save_stats=True)
