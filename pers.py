import json
import numpy as np
import octomap
import os
import trimesh

from datetime import datetime
from scipy import ndimage as ndi


def get_occupied_voxels(tree):
    res = tree.getResolution()
    occupied = []
    empty = []
    for it in tree.begin_leafs():
        center = it.getCoordinate()
        dimension = max(1, round(it.getSize() / res))
        origin = center - (dimension / 2 - 0.5) * res
        indices = np.column_stack(np.nonzero(np.ones((dimension, dimension, dimension))))
        points = origin + indices * res

        if tree.isNodeOccupied(it):
            occupied.append(points)
        else:
            empty.append(points)

    occupied = np.round(np.concatenate(occupied, axis=0), 3)
    if empty:
        empty = np.round(np.concatenate(empty, axis=0), 3)

    return occupied, empty


def save_model(tree, res, filename):
    # remove ceiling to see something
    minx, miny, minz = tree.getMetricMin()
    maxx, maxy, maxz = tree.getMetricMax()
    x_ = np.arange(minx + res / 2, maxx + res / 2, res)
    y_ = np.arange(miny + res / 2, maxy + res / 2, res)
    z_ = np.array([maxz])
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    points = np.array((x.ravel(), y.ravel(), z.ravel())).T
    for p in points:
        tree.deleteNode(p, False)

    tree.writeBinary(f'{filename}.bt'.encode())


def create_sphere(res=2, res2=2):
    points = []

    for theta in np.deg2rad(np.arange(-180, 180, res)):
        for phi in np.deg2rad(np.arange(0, 180, res2)):
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points.append((x, y, z))

    points = np.unique(np.array(points), axis=0)

    return points


class Pers:
    def __init__(self, folder, resolution=0.01, output_name="blah", lidar_poses=None, rgbd_poses=None,
                 pad_poses=None, gate_poses=None, proximity_poses=None, cam_matrices=None, cam_rotations=None,
                 proximity_rays=None, robot_inflation_value=0.5, proximity_range=10, lidar_range=10):
        self.time_stamp = datetime.now()
        self.lidar_poses = lidar_poses
        self.rgbd_poses = rgbd_poses
        self.area_poses = gate_poses
        self.proximity_poses = proximity_poses
        for pos in pad_poses:
            self.area_poses.append((pos[0], pos[1], (0.05, 0.06)))
        self.res = resolution
        self.folder = folder
        self.gt_model = octomap.OcTree(self.res)
        self.gt_model.readBinary(f"models/{folder}/model.bt".encode())
        self.occupied, self.empty = get_occupied_voxels(self.gt_model)
        self.output_name = "results/" + output_name + "/" + folder + "/"
        os.makedirs(self.output_name, exist_ok=True)

        self.human_model = octomap.OcTree(self.res)
        self.human_model.readBinary(f"models/{self.folder}/human.bt".encode())
        self.robot_model = octomap.OcTree(self.res)
        self.robot_model.readBinary(f"models/{self.folder}/robot.bt".encode())
        self.lidar_range = lidar_range
        self.rays = create_sphere(res=1, res2=1)
        self.cam_matrices = cam_matrices
        self.cam_rotations = cam_rotations
        self.proximity_rays = proximity_rays
        self.proximity_range = proximity_range
        self.robot_inflation_value = robot_inflation_value
        self.keypoints = np.loadtxt(f"models/{self.folder}/keypoints.csv", delimiter=",")

    def export_params_json(self):
        json_dict = {
            'gt_model_name': str(self.folder),
            'lidar_poses': str(self.lidar_poses),
            'rgbd_poses': str(self.rgbd_poses),
            'gate_poses': str(self.area_poses),
            'proximity_poses': str(self.proximity_poses),
            'time_stamp': self.time_stamp.strftime('%D-%H:%M')
        }
        with open(self.output_name + 'params.json', 'w') as outfile:
            json.dump(json_dict, outfile)

    def area_sensor_data(self):
        points = []
        for xlimits, ylimits, zlimits in self.area_poses:
            contactpoints = np.sum(
                (self.occupied[:, 0] >= xlimits[0]) & (self.occupied[:, 0] <= xlimits[1] + self.res) &
                (self.occupied[:, 1] >= ylimits[0]) & (self.occupied[:, 1] <= ylimits[1] + self.res) &
                (self.occupied[:, 2] >= zlimits[0]) & (self.occupied[:, 2] <= zlimits[1] + self.res))

            if contactpoints > 0:  # TODO: nafouknout kvadr - nejenom pruh ale vetsi kus odpovidajici cloveku
                x_ = np.arange(xlimits[0], xlimits[1] + self.res / 2, self.res / 2)
                y_ = np.arange(ylimits[0], ylimits[1] + self.res / 2, self.res / 2)
                z_ = np.arange(0, 3, self.res / 2)
                x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
                points.append(np.array((x.ravel(), y.ravel(), z.ravel())).T)
        return points

    def proximity_sensor_data(self):
        points = []
        for pos, rays in zip(self.proximity_poses, self.proximity_rays):
            pts = np.array(self.ray_cast(rays, pos, self.proximity_range))
            pts = np.unique(np.array(pts), axis=0)
            points.append(pts)
        return points

    def ray_cast(self, rays, pos, rng):
        points = []
        pos = np.array(pos)
        for ray in rays:
            end = np.full((3,), np.nan)
            res = self.gt_model.castRay(pos, ray, end, True, rng)
            if res:
                points.append(end)
        return points

    def detect_keypoints(self):
        for idx, pos in enumerate(self.rgbd_poses):
            rays = self.keypoints - pos
            rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
            pts = np.round(np.array(self.ray_cast(rays, pos, self.lidar_range)), 2)
            np.savetxt(f"{self.output_name}keypoints{idx}.csv", pts, fmt='%1.2f', delimiter=",")

    def lidar_sensor_data(self):
        points = [set() for _ in range(len(self.lidar_poses))]
        for pos, vis in zip(self.lidar_poses, points):
            rays = self.occupied - pos
            rays = rays[np.linalg.norm(rays, axis=1) < self.lidar_range, :]
            rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
            pts = np.array(self.ray_cast(rays, pos, self.lidar_range))
            vis |= set(zip(pts[:, 0], pts[:, 1], pts[:, 2]))  # TODO: rewrite with np.unique
        return points

    def rgbd_camera_sensor_data(self):
        visib = [set() for _ in range(len(self.rgbd_poses))]
        for pos, vis, cam_matrix, rotation in zip(self.rgbd_poses, visib, self.cam_matrices, self.cam_rotations):
            rays = self.occupied - pos
            rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
            points = []
            pos = np.array(pos)
            proj_mat = cam_matrix @ np.hstack((rotation.as_matrix().T, -rotation.as_matrix().T @ pos[:, np.newaxis]))
            for ray in rays:
                end = np.full((3,), np.nan)
                res = self.gt_model.castRay(pos, ray, end, True, self.lidar_range)
                if res:
                    proj = proj_mat @ np.array([end[0], end[1], end[2], 1])
                    proj /= proj[-1]
                    if 0 <= proj[0] <= 2 * cam_matrix[0, 2] and 0 <= proj[1] <= 2 * cam_matrix[1, 2]:
                        points.append(end)

            points = np.array(points)
            vis |= set(zip(points[:, 0], points[:, 1], points[:, 2]))
        return visib

    def proximity_robot_sensor_data(self):
        res = self.res / 2
        human_occupied, _ = get_occupied_voxels(self.human_model)
        mins = self.robot_model.getMetricMin() - self.robot_inflation_value
        dims = ((self.robot_model.getMetricSize() + 2 * self.robot_inflation_value + res) / res).astype(int)
        robot_occupied, _ = get_occupied_voxels(self.robot_model)
        coordinates = np.floor((robot_occupied - mins) / res).astype(int)

        image = np.zeros(dims, dtype=int)
        image[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = 1
        dilated = ndi.binary_dilation(image, ndi.generate_binary_structure(rank=3, connectivity=1),
                                      iterations=int(self.robot_inflation_value / res))
        coords = np.argwhere(np.logical_xor(image, dilated) == 1) * res + mins

        a = set((tuple(np.round(i, 2)) for i in coords))
        b = set((tuple(np.round(i, 2)) for i in human_occupied))
        return np.array(list(a.intersection(b)))

    def get_robot_workspace_points(self):
        robot_base = np.array([0, 0, 1])
        aabb_min = self.gt_model.getMetricMin()
        aabb_max = self.gt_model.getMetricMax()
        center = (aabb_min + aabb_max) / 2
        dimension = np.array((aabb_max - aabb_min) / self.res).astype(int)
        origin = center - dimension / 2 * self.res
        grid = np.full(dimension, -1, np.int32)
        transform = trimesh.transformations.scale_and_translate(
            scale=self.res, translate=origin
        )
        points = trimesh.voxel.VoxelGrid(encoding=grid, transform=transform).points
        # sphere around robot - maybe remove robot?
        points = points[np.linalg.norm(points - robot_base, axis=1) < 1.5, :]
        return points

    def get_human_points(self):
        aabb_min = self.human_model.getMetricMin() - np.array([0.5, 0.5, 0])
        aabb_max = self.human_model.getMetricMax() + 0.5
        center = (aabb_min + aabb_max) / 2
        dimension = np.array((aabb_max - aabb_min) / self.res).astype(int)
        origin = center - dimension / 2 * self.res
        grid = np.full(dimension, -1, np.int32)
        transform = trimesh.transformations.scale_and_translate(
            scale=self.res, translate=origin
        )
        points = trimesh.voxel.VoxelGrid(encoding=grid, transform=transform).points
        return points

    def compute_metric(self, points, covered_model, ratio=0.01):
        gt_labels = self.gt_model.getLabels(points).flatten()  # 1 occupied, -1 free
        covered_labels = covered_model.getLabels(points).flatten()  # 1 occupied, 0 free, -1 unknown
        true_occupied = np.sum((gt_labels == 1) & (covered_labels == 1))
        false_occupied = np.sum((gt_labels == 1) & (covered_labels == 0))
        unknown_occupied = np.sum((gt_labels == 1) & (covered_labels == -1))

        true_empty = np.sum((gt_labels == -1) & (covered_labels == 0))
        false_empty = np.sum((gt_labels == -1) & (covered_labels == 1))
        unknown_empty = np.sum((gt_labels == -1) & (covered_labels == -1))

        coverage_score = (true_occupied - false_occupied + ratio * true_empty - false_empty) / (
                    np.sum(gt_labels == 1) + ratio * np.sum(gt_labels == -1))
        print("(true_occupied - false_occupied + ratio * true_empty - false_empty) / (gt_occupied + ratio * gt_free)")
        print(f"({true_occupied} - {false_occupied} + {ratio} * {true_empty} - {false_empty}) / "
              f"({np.sum(gt_labels == 1)} + {ratio} * {np.sum(gt_labels == -1)})")
        print(f"Score = {coverage_score}")
        print(f"Unknown occupied = {unknown_occupied}; unknown empty = {unknown_empty}")
        return coverage_score, unknown_occupied, unknown_empty

    def compute_statistics(self, covered_model, save_stats):
        print("Robot Workspace\n---------------")
        points = self.get_robot_workspace_points()
        score_workspace = self.compute_metric(points, covered_model)
        print("\n---------------\nHuman\n---------------")
        points = self.get_human_points()
        score_human = self.compute_metric(points, covered_model)
        if save_stats:
            np.savetxt(self.output_name + "stats.csv",
                       [tuple([self.folder]) + score_workspace + score_human],
                       delimiter=", ",
                       fmt="%s")

        # TODO: compare original human bounding box and bounding box of the keypoints

    def process_sensors(self, compute_statistics=False, detect_keypoints=False, save_stats=False):
        covered_model = octomap.OcTree(self.res)

        data = self.lidar_sensor_data()
        for points, pos in zip(data, self.lidar_poses):
            covered_model.insertPointCloud(np.array(list(points)), np.array(pos), lazy_eval=True)  #
        covered_model.updateInnerOccupancy()
        save_model(covered_model, self.res, self.output_name + "lidar")

        data = self.rgbd_camera_sensor_data()
        for points, pos in zip(data, self.rgbd_poses):
            covered_model.insertPointCloud(np.array(list(points)), np.array(pos), lazy_eval=True)
        covered_model.updateInnerOccupancy()
        save_model(covered_model, self.res, self.output_name + "lidar_rgbd")

        data = self.proximity_sensor_data()
        for points, pos in zip(data, self.proximity_poses):
            covered_model.insertPointCloud(np.array(list(points)), np.array(pos), lazy_eval=True)
        covered_model.updateInnerOccupancy()
        save_model(covered_model, self.res, self.output_name + "lidar_rgbd_prox")

        data = self.area_sensor_data()
        for points in data:
            for p in points:
                covered_model.updateNode(p, True, lazy_eval=True)
        covered_model.updateInnerOccupancy()
        save_model(covered_model, self.res, self.output_name + "lidar_rgbd_prox_area")

        data = self.proximity_robot_sensor_data()
        for point in data:
            covered_model.updateNode(point, True, lazy_eval=True)  # TODO raycasting

        covered_model.updateInnerOccupancy()
        save_model(covered_model, self.res, self.output_name + "final")

        self.export_params_json()

        if detect_keypoints:
            self.detect_keypoints()
        if compute_statistics:
            self.compute_statistics(covered_model, save_stats)
