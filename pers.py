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
    free = []
    for it in tree.begin_leafs():
        center = it.getCoordinate()
        dimension = max(1, round(it.getSize() / res))
        origin = center - (dimension / 2 - 0.5) * res
        indices = np.column_stack(np.nonzero(np.ones((dimension, dimension, dimension))))
        points = origin + indices * res

        if tree.isNodeOccupied(it):
            occupied.append(points)
        else:
            free.append(points)

    occupied = np.round(np.concatenate(occupied, axis=0), 3)
    if free:
        free = np.round(np.concatenate(free, axis=0), 3)

    return occupied, free


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


def get_voxels(mins, maxs, res):
    aabb_min = mins
    aabb_max = maxs
    center = (aabb_min + aabb_max) / 2
    dimension = np.array((aabb_max - aabb_min) / res).astype(int)
    origin = center - dimension / 2 * res
    grid = np.full(dimension, 0, np.int32)
    transform = trimesh.transformations.scale_and_translate(scale=res, translate=origin)
    voxels = trimesh.voxel.VoxelGrid(encoding=grid + 1, transform=transform)  # +1 because zero cells are skipped
    return voxels, grid


class Pers:
    def __init__(self, folder, resolution=0.01, output_name="blah", lidar_poses=None, rgbd_poses=None,
                 pad_poses=None, gate_poses=None, proximity_poses=None, cam_matrices=None, cam_rotations=None,
                 proximity_rays=None, robot_inflation_value=0.5, proximity_range=10, lidar_range=10):
        self.time_stamp = datetime.now()
        self.lidar_poses = lidar_poses
        self.rgbd_poses = rgbd_poses
        self.area_poses = gate_poses
        self.proximity_poses = proximity_poses
        self.res = resolution
        for pos in pad_poses:
            self.area_poses.append((pos[0], pos[1], (self.res, 2*self.res)))
        self.folder = folder
        self.gt_model = octomap.OcTree(self.res)
        self.gt_model.readBinary(f"models/{folder}/model.bt".encode())
        self.occupied, self.free = get_occupied_voxels(self.gt_model)
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
        self.skeleton_lines = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [0, 11],
                               [0, 12], [11, 13], [12, 14], [13, 15], [14, 16]]

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
        free_points = []
        for xlimits, ylimits, zlimits in self.area_poses:
            contactpoints = np.sum(
                (self.occupied[:, 0] >= xlimits[0]) & (self.occupied[:, 0] <= xlimits[1] + self.res) &
                (self.occupied[:, 1] >= ylimits[0]) & (self.occupied[:, 1] <= ylimits[1] + self.res) &
                (self.occupied[:, 2] >= zlimits[0]) & (self.occupied[:, 2] <= zlimits[1] + self.res))

            x_ = np.arange(xlimits[0], xlimits[1] + self.res / 2, self.res / 2)
            y_ = np.arange(ylimits[0], ylimits[1] + self.res / 2, self.res / 2)
            z_ = np.arange(0, 3, self.res / 2)
            x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
            if contactpoints > 0:  # TODO: nafouknout kvadr - nejenom pruh ale vetsi kus odpovidajici cloveku
                points.append(np.array((x.ravel(), y.ravel(), z.ravel())).T)
            else:
                free_points.append(np.array((x.ravel(), y.ravel(), z.ravel())).T)
        return points, free_points

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

    def get_kpts_models(self, kpts):
        kpts_bbox = octomap.OcTree(self.res)
        for p in self.get_kpts_bbox(kpts):
            kpts_bbox.updateNode(p, True, lazy_eval=True)
        kpts_bbox.updateInnerOccupancy()
        kpts_sphere = octomap.OcTree(self.res)
        for p in self.get_kpts_sphere(kpts):
            kpts_sphere.updateNode(p, True, lazy_eval=True)
        kpts_sphere.updateInnerOccupancy()
        kpts_cyl = octomap.OcTree(self.res)
        for p in self.get_kpts_cyl(kpts):
            kpts_cyl.updateNode(p, True, lazy_eval=True)
        kpts_cyl.updateInnerOccupancy()

        return kpts_bbox, kpts_sphere, kpts_cyl

    def detect_keypoints(self):
        gt_kpts_bbox, gt_kpts_sphere, gt_kpts_cyl = self.get_kpts_models(self.keypoints)
        voxels, grid = get_voxels(self.gt_model.getMetricMin(), self.gt_model.getMetricMax(), self.res)
        gt_bbox_labels = gt_kpts_bbox.getLabels(voxels.points).flatten()  # 1 occupied, -1 free
        human_labels = self.human_model.getLabels(voxels.points).flatten()  # 1 occupied, -1 free
        gt_sphere_labels = gt_kpts_sphere.getLabels(voxels.points).flatten()  # 1 occupied, -1 free
        model_coverage_sphere = np.sum((human_labels == 1) & (gt_sphere_labels == 1)) / np.sum(
            (human_labels == 1) | (gt_sphere_labels == 1))
        gt_cyl_labels = gt_kpts_cyl.getLabels(voxels.points).flatten()  # 1 occupied, -1 free
        model_coverage_cyl = np.sum((human_labels == 1) & (gt_cyl_labels == 1)) / np.sum(
            (human_labels == 1) | (gt_cyl_labels == 1))

        scores = np.zeros((len(self.rgbd_poses), 5))
        for idx, pos in enumerate(self.rgbd_poses):
            rays = self.keypoints - pos
            rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
            kpts = np.round(np.array(self.ray_cast(rays, pos, self.lidar_range)), 2)
            np.savetxt(f"{self.output_name}keypoints{idx}.csv", kpts, fmt='%1.2f', delimiter=",")
            dists = np.linalg.norm(kpts - self.keypoints, axis=1)
            scores[idx, :2] = [np.mean(dists), np.max(dists)]

            kpts_bbox, kpts_sphere, kpts_cyl = self.get_kpts_models(kpts)
            kpts_bbox.writeBinary(f'{self.output_name}bbox_kpts{idx}.bt'.encode())
            kpts_sphere.writeBinary(f'{self.output_name}sphere_kpts{idx}.bt'.encode())
            kpts_cyl.writeBinary(f'{self.output_name}cyl_kpts{idx}.bt'.encode())
            bbox_labels = kpts_bbox.getLabels(voxels.points).flatten()  # 1 occupied, -1 free
            sphere_labels = kpts_sphere.getLabels(voxels.points).flatten()  # 1 occupied, -1 free
            cyl_labels = kpts_cyl.getLabels(voxels.points).flatten()  # 1 occupied, -1 free

            scores[idx, 2] = np.sum((gt_bbox_labels == 1) & (bbox_labels == 1)) / np.sum(
                (gt_bbox_labels == 1) | (bbox_labels == 1))  # IoU bbox
            scores[idx, 3] = np.sum((human_labels == 1) & (sphere_labels == 1)) / np.sum(
                (human_labels == 1) | (sphere_labels == 1))  # model coverage by keypoint spheres - IoU
            scores[idx, 4] = np.sum((human_labels == 1) & (cyl_labels == 1)) / np.sum(
                (human_labels == 1) | (cyl_labels == 1))  # model coverage by cylinders - IoU

        return scores

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
        voxels, grid = get_voxels(self.gt_model.getMetricMin(), self.gt_model.getMetricMax(), self.res)
        robot_base = np.array([0, 0, 1])
        
        robot_points = voxels.points[
                       (voxels.points[:, 2] >= 1) & (np.linalg.norm(voxels.points - robot_base, axis=1) < 1.5),
                       :]
        robot_labels = self.robot_model.getLabels(robot_points).flatten()  # 1 occupied, -1 free
        human_labels = self.human_model.getLabels(robot_points).flatten()  # 1 occupied, -1 free
        
        human_grid = np.copy(grid)
        occ_idx = voxels.points_to_indices(robot_points[robot_labels == 1])
        grid[occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]] = 1
        occ_idx = voxels.points_to_indices(robot_points[human_labels == 1])
        human_grid[occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]] = 1
        
        dilated = ndi.binary_dilation(grid, ndi.generate_binary_structure(rank=3, connectivity=1),
                                      iterations=int(self.robot_inflation_value / self.res))
        
        return voxels.indices_to_points(np.argwhere((dilated == 1) & (grid == 0) & (human_grid == 1))), voxels.indices_to_points(np.argwhere((dilated == 1) & (grid == 0) & (human_grid == 0)))

    def get_pers(self, covered_model):
        voxels, grid = get_voxels(self.gt_model.getMetricMin(), self.gt_model.getMetricMax(), self.res)
        robot_base = np.array([0, 0, 1])
        robot_points = voxels.points[
                       (voxels.points[:, 2] >= 1) & (np.linalg.norm(voxels.points - robot_base, axis=1) < 1.5),
                       :]  # not working without this limitation
        robot_labels = self.robot_model.getLabels(robot_points).flatten()  # 1 occupied, -1 free
        covered_labels = covered_model.getLabels(robot_points).flatten()  # 1 occupied, 0 free, -1 unknown
        covered_grid = np.copy(grid)
        occ_idx = voxels.points_to_indices(robot_points[robot_labels == 1])
        grid[occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]] = 1
        occ_idx = voxels.points_to_indices(robot_points[covered_labels == -1])
        covered_grid[occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]] = -1
        orig_grid = np.copy(grid)

        changed = True
        while changed:
            dilated = ndi.binary_dilation(grid, ndi.generate_binary_structure(rank=3, connectivity=1), iterations=1)
            idxs = np.argwhere((covered_grid == -1) & (dilated == 1) & (grid == 0))
            grid[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = 1
            changed = idxs.shape[0] > 0

        pers_model = octomap.OcTree(self.res)
        for p in voxels.indices_to_points(np.argwhere(orig_grid == 1)):
            pers_model.updateNode(p, True, lazy_eval=True)

        added_points = voxels.indices_to_points(np.argwhere((grid == 1) & (orig_grid != 1)))
        for p in added_points:
            pers_model.updateNode(p, False, lazy_eval=True)

        pers_model.updateInnerOccupancy()
        pers_model.writeBinary(f'{self.output_name}pers_model.bt'.encode())
        return added_points.shape[0]

    def get_kpts_bbox(self, kpts):
        mins = np.min(kpts, 0)
        maxs = np.max(kpts, 0)
        x_ = np.arange(mins[0], maxs[0] + self.res / 2, self.res / 2)
        y_ = np.arange(mins[1], maxs[1] + self.res / 2, self.res / 2)
        z_ = np.arange(mins[2], maxs[2], self.res / 2)
        x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
        points = np.array((x.ravel(), y.ravel(), z.ravel())).T
        return points

    def get_kpts_sphere(self, kpts, radi=3):
        points = []
        theta, phi = np.meshgrid(np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 50))
        x_ = np.sin(phi) * np.cos(theta) * radi * self.res / 2
        y_ = np.sin(phi) * np.sin(theta) * radi * self.res / 2
        z_ = np.cos(phi) * radi * self.res / 2
        for kpt in kpts:
            x = x_ + kpt[0]
            y = y_ + kpt[1]
            z = z_ + kpt[2]
            points.append(np.array((x.ravel(), y.ravel(), z.ravel())).T)
        return np.concatenate(points)

    def get_kpts_cyl(self, kpts, radi=1):
        points = []
        for line in self.skeleton_lines:
            r = radi * self.res / 2
            v = kpts[line[1]] - kpts[line[0]]
            mag = np.linalg.norm(v)
            if mag == 0:
                continue
            v = v / mag
            # make some vector not in the same direction as v
            not_v = np.array([1, 0, 0])
            if (v == not_v).all():
                not_v = np.array([0, 1, 0])
            # make vector perpendicular to v
            n1 = np.cross(v, not_v)
            if np.linalg.norm(n1) == 0:
                continue
            n1 /= np.linalg.norm(n1)
            # make unit vector perpendicular to v and n1
            n2 = np.cross(v, n1)
            # surface ranges over t from 0 to length of axis and 0 to 2*pi
            t, theta = np.meshgrid(np.linspace(0, mag, 100), np.linspace(0, 2 * np.pi, 100))
            # generate coordinates for surface
            x, y, z = [kpts[line[0]][i] + v[i] * t + r * np.sin(theta) * n1[i] + r * np.cos(theta) * n2[i] for i in
                       [0, 1, 2]]

            points.append(np.array((x.ravel(), y.ravel(), z.ravel())).T)

        return np.concatenate(points)

    def get_robot_workspace_points(self):
        robot_base = np.array([0, 0, 1])
        voxels, _ = get_voxels(self.gt_model.getMetricMin(),
                               self.gt_model.getMetricMax(), self.res)
        # sphere around robot - maybe remove robot?
        return voxels.points[np.linalg.norm(voxels.points - robot_base, axis=1) < 1.5, :]

    def get_human_points(self):
        voxels, _ = get_voxels(self.human_model.getMetricMin() - np.array([0.5, 0.5, 0]),
                               self.human_model.getMetricMax() + 0.5, self.res)
        return voxels.points

    def compute_metric(self, points, covered_model, ratio=0.01):
        gt_labels = self.gt_model.getLabels(points).flatten()  # 1 occupied, -1 free
        covered_labels = covered_model.getLabels(points).flatten()  # 1 occupied, 0 free, -1 unknown
        true_occupied = np.sum((gt_labels == 1) & (covered_labels == 1))
        false_occupied = np.sum((gt_labels == 1) & (covered_labels == 0))
        unknown_occupied = np.sum((gt_labels == 1) & (covered_labels == -1))

        true_free = np.sum((gt_labels == -1) & (covered_labels == 0))
        false_free = np.sum((gt_labels == -1) & (covered_labels == 1))
        unknown_free = np.sum((gt_labels == -1) & (covered_labels == -1))

        coverage_score = (true_occupied - false_occupied + ratio * true_free - false_free) / \
                         (np.sum(gt_labels == 1) + ratio * np.sum(gt_labels == -1))
        tpr_occupied = true_occupied / np.sum(gt_labels == 1)
        tpr_free = true_free / np.sum(gt_labels == -1)
        fpr_occupied = false_free / np.sum(gt_labels == -1)
        fpr_free = false_occupied / np.sum(gt_labels == 1)
        # print("(true_occupied - false_occupied + ratio * true_free - false_free) / (gt_occupied + ratio * gt_free)")
        # print(f"({true_occupied} - {false_occupied} + {ratio} * {true_free} - {false_free}) / "
        #       f"({np.sum(gt_labels == 1)} + {ratio} * {np.sum(gt_labels == -1)})")
        # print(f"Score = {coverage_score}")
        # print(f"Unknown occupied = {unknown_occupied}; unknown free = {unknown_free}")
        return coverage_score, unknown_occupied, unknown_free, tpr_occupied, tpr_free, fpr_occupied, fpr_free

    def compute_statistics(self, covered_model, pers_model_score, keypoints_scores, save_stats):
        print(self.folder)
        # print("Robot Workspace\n---------------")
        points = self.get_robot_workspace_points()
        score_workspace = self.compute_metric(points, covered_model)
        # print("\n---------------\nHuman\n---------------")
        points = self.get_human_points()
        score_human = self.compute_metric(points, covered_model)
        if save_stats:
            np.savetxt(self.output_name + "stats.csv",
                       [tuple([self.folder]) + score_workspace + score_human],
                       delimiter=", ",
                       fmt="%s")
            if keypoints_scores is not np.array([]):
                np.savetxt(self.output_name + "keypoint_stats.csv",
                           keypoints_scores,
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

        data, free_data = self.area_sensor_data()
        for points in data:
            for p in points:
                covered_model.updateNode(p, True, lazy_eval=True)
        for points in free_data:
            for p in points:
                covered_model.updateNode(p, False, lazy_eval=True)
        covered_model.updateInnerOccupancy()
        save_model(covered_model, self.res, self.output_name + "lidar_rgbd_prox_area")

        if self.robot_inflation_value > 0:
            data, free_data = self.proximity_robot_sensor_data()
            for point in data:
                covered_model.updateNode(point, True, lazy_eval=True)  # TODO raycasting
            for point in free_data:
                covered_model.updateNode(point, False, lazy_eval=True)
            covered_model.updateInnerOccupancy()

        save_model(covered_model, self.res, self.output_name + "final")

        voxels, grid = get_voxels(covered_model.getMetricMin(), covered_model.getMetricMax(), self.res)

        covered_model_visu = octomap.OcTree(self.res)
        covered_labels = covered_model.getLabels(voxels.points).flatten()  # 1 occupied, 0 free, -1 unknown
        for point in voxels.points[covered_labels == 1, :]:
            covered_model_visu.updateNode(point, True, lazy_eval=True)
        for point in voxels.points[covered_labels == -1, :]:
            covered_model_visu.updateNode(point, True, lazy_eval=True)
        for point in voxels.points[covered_labels == 0, :]:
            covered_model_visu.updateNode(point, False, lazy_eval=True)
        covered_model_visu.updateInnerOccupancy()
        save_model(covered_model_visu, self.res, self.output_name + "unknown_occupied")

        self.export_params_json()
        pers_model_score = self.get_pers(covered_model)
        keypoints_scores = np.array([])
        if detect_keypoints:
            keypoints_scores = self.detect_keypoints()
        print(pers_model_score, keypoints_scores)
        if compute_statistics:
            self.compute_statistics(covered_model, pers_model_score, keypoints_scores, save_stats)
