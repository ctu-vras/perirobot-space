import json
import numpy as np
import octomap
import os
import trimesh

from datetime import datetime
from scipy import ndimage as ndi


def get_kpts_models(res, kpts, output_name=""):
    occ_kpts_bbox = None
    occ_kpts_cyl = None
    occ_kpts_sphere = None
    if output_name != "":
        occ_kpts_bbox = octomap.OcTree(res)
        occ_kpts_sphere = octomap.OcTree(res)
        occ_kpts_cyl = octomap.OcTree(res)
        occ_kpts_bbox.readBinary(f"{output_name}/temp.bt".encode())
        occ_kpts_sphere.readBinary(f"{output_name}/temp.bt".encode())
        occ_kpts_cyl.readBinary(f"{output_name}/temp.bt".encode())

        if kpts.shape[0] > 0:
            for p in get_kpts_bbox(res, kpts):
                occ_kpts_bbox.updateNode(p, True, lazy_eval=True)
            occ_kpts_bbox.updateInnerOccupancy()
            for p in get_kpts_sphere(res, kpts):
                occ_kpts_sphere.updateNode(p, True, lazy_eval=True)
            occ_kpts_sphere.updateInnerOccupancy()
            for p in get_kpts_cyl(res, kpts):
                occ_kpts_cyl.updateNode(p, True, lazy_eval=True)
            occ_kpts_cyl.updateInnerOccupancy()

    kpts_bbox = octomap.OcTree(res)
    kpts_sphere = octomap.OcTree(res)
    kpts_cyl = octomap.OcTree(res)
    if kpts.shape[0] > 0:
        for p in get_kpts_bbox(res, kpts):
            kpts_bbox.updateNode(p, True, lazy_eval=True)
        kpts_bbox.updateInnerOccupancy()
        for p in get_kpts_sphere(res, kpts):
            kpts_sphere.updateNode(p, True, lazy_eval=True)
        kpts_sphere.updateInnerOccupancy()
        for p in get_kpts_cyl(res, kpts):
            kpts_cyl.updateNode(p, True, lazy_eval=True)
        kpts_cyl.updateInnerOccupancy()

    return kpts_bbox, kpts_sphere, kpts_cyl, occ_kpts_bbox, occ_kpts_sphere, occ_kpts_cyl


def get_kpts_bbox(res, kpts):
    mins = np.min(kpts, 0)
    maxs = np.max(kpts, 0)
    x_ = np.arange(mins[0], maxs[0] + res / 2, res / 2)
    y_ = np.arange(mins[1], maxs[1] + res / 2, res / 2)
    z_ = np.arange(mins[2], maxs[2], res / 2)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    points = np.array((x.ravel(), y.ravel(), z.ravel())).T
    return points


def get_kpts_sphere(res, kpts, radi=8):
    points = []
    theta, phi = np.meshgrid(np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 50))
    x_ = np.sin(phi) * np.cos(theta) * radi * res / 2
    y_ = np.sin(phi) * np.sin(theta) * radi * res / 2
    z_ = np.cos(phi) * radi * res / 2
    for kpt in kpts:
        x = x_ + kpt[0]
        y = y_ + kpt[1]
        z = z_ + kpt[2]
        points.append(np.array((x.ravel(), y.ravel(), z.ravel())).T)
    return np.concatenate(points)


def get_kpts_cyl(res, kpts, radi=10):
    skeleton_lines = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [6, 8], [7, 9],
                      [8, 10], [0, 11], [0, 12], [11, 13], [12, 14], [13, 15], [14, 16]]
    points = []
    for line in skeleton_lines:
        for i in range(5):
            r = radi * res / 2 * (i + 1) / 5
            if line[1] >= kpts.shape[0] or line[0] >= kpts.shape[0]:
                continue
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
    if len(points) == 0:
        return []
    return np.concatenate(points)


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


def save_model(tree, gt_model, res, filename):
    # remove ceiling to see something
    minx, miny, minz = gt_model.getMetricMin()
    maxx, maxy, maxz = gt_model.getMetricMax()
    x_ = np.arange(minx + res / 2, maxx + res / 2, res)
    y_ = np.arange(miny + res / 2, maxy + res / 2, res)
    z_ = np.arange(maxz - 5 * res / 2, maxz + 5 * res / 2, res)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    points = np.array((x.ravel(), y.ravel(), z.ravel())).T
    for p in points:
        tree.deleteNode(p, False)

    tree.writeBinary(f'{filename}.bt'.encode())


def create_sphere(min_ang=135, max_ang=180, res=0.7, res2=0.7, rot_matrix=None):
    points = []
    for theta in np.deg2rad(np.arange(-180, 180, res)):
        for phi in np.deg2rad(np.arange(min_ang, max_ang, res2)):
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points.append((x, y, z))

    points = np.unique(np.array(points), axis=0)
    if rot_matrix is not None:
        points = (rot_matrix.as_matrix() @ points.T).T

    return points


def create_rays(min_ang=-15 + 90, res=0.25, res2=2.5, rot_matrix=None):
    points = []
    for i in range(10):
        theta = np.deg2rad(min_ang + res2 * i)
        for phi in np.deg2rad(np.arange(0, 180, res)):
            x = np.sin(phi) * np.cos(theta)
            y = np.cos(phi)
            z = np.sin(phi) * np.sin(theta)
            points.append((x, y, z))

    points = np.unique(np.array(points), axis=0)
    if rot_matrix is not None:
        points = (rot_matrix.as_matrix() @ points.T).T

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


class Sensor:
    def __init__(self, poses, res):
        self.poses = poses
        self.res = res

    def get_measurement(self):
        pass


class RangeSensor(Sensor):
    def __init__(self, poses, res, gt_model, noise_level, rng, min_dist):
        super().__init__(poses, res)
        self.gt_model = gt_model
        self.noise_level = noise_level
        self.rng = rng
        self.min_dist = min_dist

    def ray_cast(self, rays, pos, rng):
        points = []
        pos = np.array(pos)
        for ray in rays:
            end = np.full((3,), np.nan)
            res = self.gt_model.castRay(pos, ray, end, True, rng)
            if res:
                dist = np.linalg.norm(end - pos)
                if dist > self.min_dist:
                    if self.noise_level:
                        norm = dist + np.random.normal(0, self.noise_level)
                        end = self.gt_model.keyToCoord(self.gt_model.coordToKey(pos + ray * norm))
                    points.append(end)
        return points

    def get_measurement(self):
        pass


class ZoneSensor(Sensor):
    def __init__(self, poses, res, zones, occupied):
        super().__init__(poses, res)
        self.zones = zones
        self.occupied = occupied

    def get_measurement(self):
        points = []
        free_points = []
        for (xlimits, ylimits, zlimits), (xzone, yzone, zzone) in zip(self.poses, self.zones):
            contactpoints = np.sum(
                (self.occupied[:, 0] >= xlimits[0]) & (self.occupied[:, 0] <= xlimits[1] + self.res) &
                (self.occupied[:, 1] >= ylimits[0]) & (self.occupied[:, 1] <= ylimits[1] + self.res) &
                (self.occupied[:, 2] >= zlimits[0]) & (self.occupied[:, 2] <= zlimits[1] + self.res))

            x_ = np.arange(xzone[0], xzone[1] + self.res / 2, self.res / 2)
            y_ = np.arange(yzone[0], yzone[1] + self.res / 2, self.res / 2)
            z_ = np.arange(0, 3, self.res / 2)
            x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
            if contactpoints > 0:
                points.append(np.array((x.ravel(), y.ravel(), z.ravel())).T)
            else:
                free_points.append(np.array((x.ravel(), y.ravel(), z.ravel())).T)
        return points, free_points


class AreaSensor(Sensor):
    def __init__(self, poses, res, occupied):
        super().__init__(poses, res)
        self.occupied = occupied

    def get_measurement(self):
        points = []
        free_points = []
        for xlimits, ylimits, zlimits in self.poses:
            contactpoints = np.sum(
                (self.occupied[:, 0] >= xlimits[0]) & (self.occupied[:, 0] <= xlimits[1] + self.res) &
                (self.occupied[:, 1] >= ylimits[0]) & (self.occupied[:, 1] <= ylimits[1] + self.res) &
                (self.occupied[:, 2] >= zlimits[0]) & (self.occupied[:, 2] <= zlimits[1] + self.res))

            x_ = np.arange(xlimits[0], xlimits[1] + self.res / 2, self.res / 2)
            y_ = np.arange(ylimits[0], ylimits[1] + self.res / 2, self.res / 2)
            z_ = np.arange(0, 3, self.res / 2)
            x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
            if contactpoints > 0:
                points.append(np.array((x.ravel(), y.ravel(), z.ravel())).T)
            else:
                free_points.append(np.array((x.ravel(), y.ravel(), z.ravel())).T)
        return points, free_points


class Lidar(RangeSensor):
    def __init__(self, poses, res, gt_model, noise, rng, min_dist, rot_matrix):
        super().__init__(poses, res, gt_model, noise, rng, min_dist)
        if rng < 30:
            self.rays = create_sphere(min_ang=0, max_ang=55, rot_matrix=rot_matrix)
        else:
            self.rays = create_rays(rot_matrix=rot_matrix)

    def get_measurement(self):
        points = [set() for _ in range(len(self.poses))]
        for pos, vis in zip(self.poses, points):
            rays = self.rays
            pts = np.array(self.ray_cast(rays, pos, self.rng))
            vis |= set(zip(pts[:, 0], pts[:, 1], pts[:, 2]))
        return points


class RGBD(RangeSensor):
    def __init__(self, poses, res, gt_model, noise, occupied, rng, min_dist, cam_mat, cam_rot, keypoints):
        super().__init__(poses, res, gt_model, noise, rng, min_dist)
        self.occupied = occupied
        self.cam_mat = cam_mat
        self.cam_rot = cam_rot
        self.keypoints = keypoints

    def get_measurement(self):
        visib = [set() for _ in range(len(self.poses))]
        for pos, vis, cam_matrix, rotation in zip(self.poses, visib, self.cam_mat, self.cam_rot):
            rays = self.occupied - pos
            rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
            points = []
            pos = np.array(pos)
            proj_mat = cam_matrix @ np.hstack((rotation.as_matrix().T, -rotation.as_matrix().T @ pos[:, np.newaxis]))
            depth_im = np.zeros((2 * int(cam_matrix[0, 2]), 2 * int(cam_matrix[1, 2]))) + 10000
            closest_points = np.full((3, 2 * int(cam_matrix[0, 2]), 2 * int(cam_matrix[1, 2])), np.nan)
            for ray in rays:
                end = np.full((3,), np.nan)
                res = self.gt_model.castRay(pos, ray, end, True, self.rng)
                if res:
                    dist = np.linalg.norm(end - pos)
                    if dist > self.min_dist:
                        proj = proj_mat @ np.array([end[0], end[1], end[2], 1])
                        proj /= proj[-1]
                        if self.noise_level:
                            norm = dist + np.random.normal(0, self.noise_level)
                            end = self.gt_model.keyToCoord(self.gt_model.coordToKey(pos + ray * norm))
                            dist = np.linalg.norm(end - pos)
                        if 0 <= proj[0] <= 2 * cam_matrix[0, 2] and 0 <= proj[1] <= 2 * cam_matrix[1, 2] \
                                and depth_im[int(proj[0]), int(proj[1])] > dist:
                            depth_im[int(proj[0]), int(proj[1])] = dist
                            closest_points[:, int(proj[0]), int(proj[1])] = end

            for i in range(2 * int(cam_matrix[0, 2])):
                for j in range(2 * int(cam_matrix[1, 2])):
                    if not np.isnan(closest_points[0, i, j]):
                        points.append(closest_points[:, i, j])
            points = np.array(points)
            vis |= set(zip(points[:, 0], points[:, 1], points[:, 2]))
        return visib

    def detect_kpts(self, output_name, human_model):
        gt_kpts_bbox, gt_kpts_sphere, gt_kpts_cyl, _, _, _ = get_kpts_models(self.res, self.keypoints)
        voxels, grid = get_voxels(self.gt_model.getMetricMin(), self.gt_model.getMetricMax(), self.res)
        gt_bbox_labels = gt_kpts_bbox.getLabels(voxels.points).flatten()  # 1 occupied, -1 free
        human_labels = human_model.getLabels(voxels.points).flatten()  # 1 occupied, -1 free
        kpts_cyls = []
        kpts_bboxes = []
        kpts_spheres = []
        scores = np.zeros((len(self.poses), 5))
        for idx, (pos, cam_matrix, rotation) in enumerate(zip(self.poses, self.cam_mat, self.cam_rot)):
            rays = self.keypoints - pos
            rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
            pos = np.array(pos)
            proj_mat = cam_matrix @ np.hstack((rotation.as_matrix().T, -rotation.as_matrix().T @ pos[:, np.newaxis]))
            kpts = np.full_like(self.keypoints, np.inf)
            for i, ray in enumerate(rays):
                end = np.full((3,), np.nan)
                res = self.gt_model.castRay(pos, ray, end, True, self.rng)
                if res:
                    if self.noise_level:
                        dist = np.linalg.norm(end - pos)
                        norm = dist + np.random.normal(0, self.noise_level)
                        end = self.gt_model.keyToCoord(self.gt_model.coordToKey(pos + ray * norm))

                    proj = proj_mat @ np.array([end[0], end[1], end[2], 1])
                    proj /= proj[-1]
                    if 0 <= proj[0] <= 2 * cam_matrix[0, 2] and 0 <= proj[1] <= 2 * cam_matrix[1, 2]:
                        kpts[i, :] = end
            kpts = np.array(kpts)
            if kpts.shape[0] == 0:
                return scores
            np.savetxt(f"{output_name}keypoints{idx}.csv", kpts, fmt='%1.2f', delimiter=",")
            dists = np.linalg.norm(kpts - self.keypoints, axis=1)
            print(dists)
            scores[idx, :2] = [np.mean(dists), np.max(dists)]
            kpts = kpts[kpts[:, 1] != np.inf, :]
            kpts_bbox, kpts_sphere, kpts_cyl, occ_kpts_bbox, occ_kpts_sphere, occ_kpts_cyl = \
                get_kpts_models(self.res, kpts, output_name)
            kpts_bbox.writeBinary(f'{output_name}bbox_kpts{idx}.bt'.encode())
            kpts_sphere.writeBinary(f'{output_name}sphere_kpts{idx}.bt'.encode())
            kpts_cyl.writeBinary(f'{output_name}cyl_kpts{idx}.bt'.encode())
            bbox_labels = kpts_bbox.getLabels(voxels.points).flatten()  # 1 occupied, -1 free
            sphere_labels = kpts_sphere.getLabels(voxels.points).flatten()  # 1 occupied, -1 free
            cyl_labels = kpts_cyl.getLabels(voxels.points).flatten()  # 1 occupied, -1 free

            scores[idx, 2] = np.sum((gt_bbox_labels == 1) & (bbox_labels == 1)) / np.sum(
                (gt_bbox_labels == 1) | (bbox_labels == 1))  # IoU bbox
            scores[idx, 3] = np.sum((human_labels == 1) & (sphere_labels == 1)) / np.sum(
                (human_labels == 1) | (sphere_labels == 1))  # model coverage by keypoint spheres - IoU
            scores[idx, 4] = np.sum((human_labels == 1) & (cyl_labels == 1)) / np.sum(
                (human_labels == 1) | (cyl_labels == 1))  # model coverage by cylinders - IoU
            kpts_cyls.append(occ_kpts_cyl)
            kpts_bboxes.append(occ_kpts_bbox)
            kpts_spheres.append(occ_kpts_sphere)

        return scores, kpts_spheres[0], kpts_cyls[0], kpts_bboxes[0]


class RobotProximity(Sensor):
    def __init__(self, res, gt_model, noise, infl, robot_model, human_model):
        super().__init__([], res)
        self.gt_model = gt_model
        self.noise_level = noise
        self.infl_value = infl
        self.robot_model = robot_model
        self.human_model = human_model

    def get_measurement(self):
        voxels, grid = get_voxels(self.gt_model.getMetricMin(), self.gt_model.getMetricMax(), self.res)
        robot_base = np.array([0, 0, 1])

        robot_points = voxels.points[
                       (voxels.points[:, 2] >= 1) & (np.linalg.norm(voxels.points - robot_base, axis=1) < 1.5),
                       :]
        robot_labels = self.robot_model.getLabels(robot_points).flatten()  # 1 occupied, -1 free
        human_labels = self.gt_model.getLabels(robot_points).flatten()  # 1 occupied, -1 free

        human_grid = np.copy(grid)
        occ_idx = voxels.points_to_indices(robot_points[robot_labels == 1])
        grid[occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]] = 1
        occ_idx = voxels.points_to_indices(robot_points[human_labels == 1])
        human_grid[occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2]] = 1

        dilated = ndi.binary_dilation(grid, ndi.generate_binary_structure(rank=3, connectivity=1),
                                      iterations=int(self.infl_value / self.res))

        cover_points = voxels.indices_to_points(np.argwhere((dilated == 1) & (grid == 0)))
        occupied_points = voxels.indices_to_points(np.argwhere((dilated == 1) & (grid == 0) & (human_grid == 1)))
        occupied_points += np.random.normal(0, self.noise_level, occupied_points.shape)

        return occupied_points, cover_points


class Proximity(RangeSensor):
    def __init__(self, poses, res, gt_model, rays, noise, rng, min_dist):
        super().__init__(poses, res, gt_model, noise, rng, min_dist)
        self.rays = rays

    def get_measurement(self):
        points = []
        if not self.poses or not self.rays:
            return points
        for pos, rays in zip(self.poses, self.rays):
            pts = np.array(self.ray_cast(rays, pos, self.rng))
            pts = np.unique(np.array(pts), axis=0)
            points.append(pts)
        return points


class Camera(ZoneSensor):
    def __init__(self, poses, res, zones, occupied, gt_model, cam_mat, cam_rot, keypoints, rng):
        super().__init__(poses, res, zones, occupied)
        self.keypoints = keypoints
        self.gt_model = gt_model
        self.rng = rng
        self.cam_mat = cam_mat
        self.cam_rot = cam_rot

    def get_measurement(self):
        points = []
        free_points = []

        for idx, (pos, (xzone, yzone, zzone), cam_matrix, rotation) in enumerate(
                zip(self.poses, self.zones, self.cam_mat, self.cam_rot)):
            rays = self.keypoints - pos
            rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
            pos = np.array(pos)
            proj_mat = cam_matrix @ np.hstack((rotation.as_matrix().T, -rotation.as_matrix().T @ pos[:, np.newaxis]))
            keypoints_num = 0
            for i, ray in enumerate(rays):
                end = np.full((3,), np.nan)
                res = self.gt_model.castRay(pos, ray, end, True, self.rng)
                if res:
                    proj = proj_mat @ np.array([end[0], end[1], end[2], 1])
                    proj /= proj[-1]
                    if 0 <= proj[0] <= 2 * cam_matrix[0, 2] and 0 <= proj[1] <= 2 * cam_matrix[1, 2]:
                        if self.gt_model.coordToKey(end) == self.gt_model.coordToKey(self.keypoints[i]):
                            keypoints_num += 1

            x_ = np.arange(xzone[0], xzone[1] + self.res / 2, self.res / 2)
            y_ = np.arange(yzone[0], yzone[1] + self.res / 2, self.res / 2)
            z_ = np.arange(0, 3, self.res / 2)
            x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
            if keypoints_num > 0:
                points.append(np.array((x.ravel(), y.ravel(), z.ravel())).T)
            else:
                free_points.append(np.array((x.ravel(), y.ravel(), z.ravel())).T)
        return points, free_points


class Gate(ZoneSensor):
    def __init__(self, poses, res, zones, occupied):
        super().__init__(poses, res, zones, occupied)

    def get_measurement(self):
        return super().get_measurement()


class Pad(AreaSensor):
    def __init__(self, poses, res, occupied):
        super().__init__(poses, res, occupied)


class Pers:
    def __init__(self, folder, resolution=0.01, output_name="blah", lidar_poses=None, rgbd_poses=None, pad_poses=None,
                 gate_poses=None, prox_poses=None, cam_poses=None, cam_matrices=None, cam_rotations=None,
                 prox_rays=None, robot_inflation_value=0.0, prox_range=10, lidar_range=10, rgbd_range=10,
                 noise_lidar=0.1, noise_rgbd=0.1, noise_prox=0.1, noise_rob_prox=0.1, lidar_mindist=0.2,
                 rgbd_mindist=0.3, prox_mindist=0.1, robot_zone=None, add_robot_vol=False, dynamic_model=-1):
        self.time_stamp = datetime.now()
        self.res = resolution
        self.folder = folder
        self.gt_model = octomap.OcTree(self.res)
        if dynamic_model > -1:
            self.gt_model.readBinary(f"models/{folder}/model{dynamic_model}.bt".encode())
        else:
            self.gt_model.readBinary(f"models/{folder}/model.bt".encode())
        self.occupied, self.free = get_occupied_voxels(self.gt_model)
        self.output_name = "results/" + output_name + "/" + folder + "/"
        os.makedirs(self.output_name, exist_ok=True)
        try:
            self.human_model = octomap.OcTree(self.res)
            self.human_model.readBinary(f"models/{self.folder}/human.bt".encode())
        except Exception:
            self.human_model = None
        self.robot_model = octomap.OcTree(self.res)
        self.robot_model.readBinary(f"models/{self.folder}/robot.bt".encode())
        self.add_robot_vol = add_robot_vol
        self.lidars = None
        self.rgbd = None
        self.gates = None
        self.pads = None
        self.proximity = None
        self.rob_prox = None
        self.cameras = None
        if lidar_poses is not None:
            self.lidars = Lidar(lidar_poses, self.res, self.gt_model, noise_lidar, lidar_range, lidar_mindist,
                                cam_rotations[0])
        if rgbd_poses is not None:
            keypoints = np.loadtxt(f"models/{folder}/keypoints.csv", delimiter=",")
            self.rgbd = RGBD(rgbd_poses, self.res, self.gt_model, noise_rgbd, self.occupied,
                             rgbd_range, rgbd_mindist, cam_matrices, cam_rotations, keypoints)
        if gate_poses is not None:
            self.gates = Gate(gate_poses, self.res, len(gate_poses) * robot_zone, self.occupied)
        if pad_poses is not None:
            self.pads = Pad([(p[0], p[1], (self.res, 2 * self.res)) for p in pad_poses], self.res, self.occupied)
        if prox_poses is not None:
            self.proximity = Proximity(prox_poses, self.res, self.gt_model, noise_prox, prox_rays, prox_range,
                                       prox_mindist)
        if robot_inflation_value > 0:
            self.rob_prox = RobotProximity(self.res, self.gt_model, noise_rob_prox, robot_inflation_value,
                                           self.robot_model, self.human_model)
        if cam_poses is not None:
            keypoints = np.loadtxt(f"models/{folder}/keypoints.csv", delimiter=",")
            self.cameras = Camera(cam_poses, self.res, len(cam_poses) * robot_zone, self.occupied, self.gt_model,
                                  cam_matrices, cam_rotations, keypoints, lidar_range)

    def export_params_json(self):
        json_dict = {
            'gt_model_name': str(self.folder),
            'lidar_poses': str(self.lidars.poses),
            'rgbd_poses': str(self.rgbd.poses),
            'gate_poses': str(self.gates.poses),
            'proximity_poses': str(self.proximity.poses),
            'time_stamp': self.time_stamp.strftime('%D-%H:%M')
        }
        with open(self.output_name + 'params.json', 'w') as outfile:
            json.dump(json_dict, outfile)

    def get_robot_workspace_points(self):
        robot_base = np.array([0, 0, 1])
        voxels, _ = get_voxels(self.gt_model.getMetricMin(),
                               self.gt_model.getMetricMax(), self.res)
        # sphere around robot - maybe remove robot?
        return voxels.points[np.linalg.norm(voxels.points - robot_base, axis=1) < 1.5, :]

    def get_human_points(self):
        voxels, _ = get_voxels(self.human_model.getMetricMin() - np.array([0.3, 0.3, 0.0]),
                               self.human_model.getMetricMax() + np.array([0.3, 0.0, 0.3]), self.res)
        return voxels.points

    def compute_metric(self, points, covered_model):
        gt_labels = self.gt_model.getLabels(points).flatten()  # 1 occupied, -1 free
        covered_labels = covered_model.getLabels(points).flatten()  # 1 occupied, 0 free, -1 unknown
        true_occupied = np.sum((gt_labels == 1) & (covered_labels == 1))
        false_occupied = np.sum((gt_labels == 1) & (covered_labels == 0))
        unknown_occupied = np.sum((gt_labels == 1) & (covered_labels == -1))

        covered_occupied = np.sum(covered_labels == 1)
        covered_free = np.sum(covered_labels == 0)
        covered_unknown = np.sum(covered_labels == -1)
        gt_occupied = np.sum(gt_labels == 1)
        gt_free = np.sum(gt_labels == -1)

        true_free = np.sum((gt_labels == -1) & (covered_labels == 0))
        false_free = np.sum((gt_labels == -1) & (covered_labels == 1))
        unknown_free = np.sum((gt_labels == -1) & (covered_labels == -1))
        f1_score = 2 * true_occupied / (
                    2 * true_occupied + false_occupied + false_free + unknown_free + unknown_occupied)

        # copied from sklearn library
        confusion = np.array([[true_free, false_free, unknown_free],
                              [false_occupied, true_occupied, unknown_occupied],
                              [0, 0, 0]])
        n_classes = confusion.shape[0]
        sum0 = np.sum(confusion, axis=0)
        sum1 = np.sum(confusion, axis=1)
        expected = np.outer(sum0, sum1) / np.sum(sum0)

        w_mat = np.ones([n_classes, n_classes], dtype=int)
        w_mat.flat[:: n_classes + 1] = 0

        kappa = 1 - np.sum(w_mat * confusion) / np.sum(w_mat * expected)
        return true_occupied, false_occupied, unknown_occupied, true_free, false_free, unknown_free, covered_occupied, \
            covered_free, covered_unknown, gt_occupied, gt_free, f1_score, kappa

    def compute_statistics(self, covered_model, keypoints_scores, save_stats,
                           kpts_sphere=None, kpts_cyl=None, kpts_bbox=None):
        print(self.folder)

        points = self.get_robot_workspace_points()
        score_ws = self.compute_metric(points, covered_model)
        score_ws_kpts_sp = tuple(13 * [-1])
        if kpts_sphere is not None:
            score_ws_kpts_sp = self.compute_metric(points, kpts_sphere)
        score_ws_kpts_cyl = tuple(13 * [-1])
        if kpts_cyl is not None:
            score_ws_kpts_cyl = self.compute_metric(points, kpts_cyl)
        score_ws_kpts_bbox = tuple(13 * [-1])
        if kpts_bbox is not None:
            score_ws_kpts_bbox = self.compute_metric(points, kpts_bbox)

        points = self.get_human_points()
        score_h = self.compute_metric(points, covered_model)
        score_h_kpts_sp = tuple(13 * [-1])
        if kpts_sphere is not None:
            score_h_kpts_sp = self.compute_metric(points, kpts_sphere)
        score_h_kpts_cyl = tuple(13 * [-1])
        if kpts_cyl is not None:
            score_h_kpts_cyl = self.compute_metric(points, kpts_cyl)
        score_h_kpts_bbox = tuple(13 * [-1])
        if kpts_bbox is not None:
            score_h_kpts_bbox = self.compute_metric(points, kpts_bbox)

        if save_stats:
            np.savetxt(self.output_name + "stats.csv",
                       [tuple([self.folder]) + score_ws + score_h + score_ws_kpts_sp + score_h_kpts_sp +
                        score_ws_kpts_cyl + score_h_kpts_cyl + score_ws_kpts_bbox + score_h_kpts_bbox],
                       delimiter=", ",
                       fmt="%s")
            if keypoints_scores is not np.array([]):
                np.savetxt(self.output_name + "keypoint_stats.csv",
                           keypoints_scores,
                           delimiter=", ",
                           fmt="%s")

    def process_sensors(self, compute_statistics=False, detect_keypoints=False, save_stats=False):
        covered_model = octomap.OcTree(self.res)
        if self.lidars:
            data = self.lidars.get_measurement()
            for points, pos in zip(data, self.lidars.poses):
                covered_model.insertPointCloud(np.array(list(points)), np.array(pos), lazy_eval=True)  #
            covered_model.updateInnerOccupancy()
            save_model(covered_model, self.gt_model, self.res, self.output_name + "lidar")
        if self.rgbd:
            data = self.rgbd.get_measurement()
            for points, pos in zip(data, self.rgbd.poses):
                covered_model.insertPointCloud(np.array(list(points)), np.array(pos), lazy_eval=True)
            covered_model.updateInnerOccupancy()
            save_model(covered_model, self.gt_model, self.res, self.output_name + "lidar_rgbd")
        if self.proximity:
            data = self.proximity.get_measurement()
            for points, pos in zip(data, self.proximity.get_measurement()):
                covered_model.insertPointCloud(np.array(list(points)), np.array(pos), lazy_eval=True)
            covered_model.updateInnerOccupancy()
            save_model(covered_model, self.gt_model, self.res, self.output_name + "lidar_rgbd_prox")
        if self.gates:
            data, free_data = self.gates.get_measurement()
            for points in data:
                for p in points:
                    covered_model.updateNode(p, True, lazy_eval=True)
            for points in free_data:
                for p in points:
                    covered_model.updateNode(p, False, lazy_eval=True)
            covered_model.updateInnerOccupancy()
            save_model(covered_model, self.gt_model, self.res, self.output_name + "lidar_rgbd_prox_area")
        if self.cameras:
            data, free_data = self.cameras.get_measurement()
            for points in data:
                for p in points:
                    covered_model.updateNode(p, True, lazy_eval=True)
            for points in free_data:
                for p in points:
                    covered_model.updateNode(p, False, lazy_eval=True)
            covered_model.updateInnerOccupancy()
            save_model(covered_model, self.gt_model, self.res, self.output_name + "lidar_rgbd_prox_area_cam")
        if self.pads:
            data, free_data = self.pads.get_measurement()
            for points in data:
                for p in points:
                    covered_model.updateNode(p, True, lazy_eval=True)
            for points in free_data:
                for p in points:
                    covered_model.updateNode(p, False, lazy_eval=True)
            covered_model.updateInnerOccupancy()
            save_model(covered_model, self.gt_model, self.res, self.output_name + "lidar_rgbd_prox_area_cam_pads")

        if self.rob_prox:  # TODO: change to ray casting
            data, free_data = self.rob_prox.get_measurement()
            for point in data:
                covered_model.updateNode(point, True, lazy_eval=True)
            for point in free_data:
                covered_model.updateNode(point, False, lazy_eval=True)
            covered_model.updateInnerOccupancy()

        if self.add_robot_vol:
            robot_points, _ = get_occupied_voxels(self.robot_model)
            for point in robot_points:
                covered_model.updateNode(point, True, lazy_eval=True)

        save_model(covered_model, self.gt_model, self.res, self.output_name + "final")

        voxels, grid = get_voxels(covered_model.getMetricMin(), covered_model.getMetricMax(), self.res)

        covered_model_visual = octomap.OcTree(self.res)
        covered_labels = covered_model.getLabels(voxels.points).flatten()  # 1 occupied, 0 free, -1 unknown
        for point in voxels.points[covered_labels == 1, :]:
            covered_model_visual.updateNode(point, True, lazy_eval=True)
        for point in voxels.points[covered_labels == -1, :]:
            covered_model_visual.updateNode(point, True, lazy_eval=True)
        for point in voxels.points[covered_labels == 0, :]:
            covered_model_visual.updateNode(point, False, lazy_eval=True)
        covered_model_visual.updateInnerOccupancy()
        save_model(covered_model_visual, self.gt_model, self.res, self.output_name + "unknown_occupied")

        # self.export_params_json()
        keypoints_scores = np.array([])
        kpts_sphere = None
        kpts_cyl = None
        kpts_bbox = None
        if detect_keypoints:
            covered_model.writeBinary(f'{self.output_name}temp.bt'.encode())
            keypoints_scores, kpts_sphere, kpts_cyl, kpts_bbox = self.rgbd.detect_kpts(self.output_name,
                                                                                       self.human_model)

        if compute_statistics:
            self.compute_statistics(covered_model, keypoints_scores, save_stats, kpts_sphere, kpts_cyl, kpts_bbox)
