import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import octomap

from scipy.spatial.transform import Rotation
from scipy import ndimage as ndi

RESOLUTION = 0.05  # m resolution of octomap
LASER_RANGE = 10


def plotVisibility3D(tree, converted_poses, rng, rays=None, occupied=None):
    colors = []
    index = 0
    p2 = []

    for pos in converted_poses:
        if occupied is not None:
            rays = occupied - pos
            rays = rays[np.linalg.norm(rays, axis=1) < rng, :]
            rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        points = rayCast3D(tree, rays, pos, rng)
        p2 += points
        colors += [index for _ in range(len(points))]
        index += 1

    p2 += converted_poses
    colors += [index for _ in range(len(converted_poses))]
    p2 = np.array(p2)
    p, indices = np.unique(p2, axis=0, return_index=True)
    colors = np.array(colors)
    splot = go.Scatter3d(
        x=p2[indices, 0].flatten(),
        y=p2[indices, 1].flatten(),
        z=p2[indices, 2].flatten(),
        mode='markers',
        marker=dict(color=colors[indices],
                    size=5, opacity=1, colorscale=[
                [0, "rgb(31,120,180)"],
                [0.25, "rgb(178,223,138)"],
                [0.5, "rgb(51,160,44)"],
                [0.75, "rgb(251,154,153)"],
                [1, "rgb(227,26,28)"]])
    )

    lay = go.Layout(autosize=True, scene=dict(camera=dict(eye=dict(x=1.15, y=1.15, z=0.8)), aspectmode='data'))
    fig = go.Figure(data=[splot], layout=lay)
    fig.show()


def plotSkeleton(tree, pos, keypoints, rng):
    skeleton_lines = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [0, 11], [0, 12],
                      [11, 13], [12, 14], [13, 15], [14, 16]]
    rays = keypoints - pos
    rays = rays[np.linalg.norm(rays, axis=1) < rng, :]
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    points = np.array(rayCast3D(tree, rays, pos, rng))
    print(points)

    lay = go.Layout(autosize=True, scene=dict(camera=dict(eye=dict(x=1.15, y=1.15, z=0.8)), aspectmode='data'))
    fig = go.Figure(layout=lay)
    fig.add_trace(
        go.Scatter3d(x=points[:, 0].flatten(), y=points[:, 1].flatten(), z=points[:, 2].flatten(), mode='markers'))

    for l in skeleton_lines:
        fig.add_trace(
            go.Scatter3d(x=points[l, 0].flatten(), y=points[l, 1].flatten(), z=points[l, 2].flatten(), mode='lines'))

    fig.show()


def plotPadData(poses, occupied):
    colors = []
    index = 0
    p2 = []

    for xlimits, ylimits in poses:
        contactPoints = np.sum(
            (occupied[:, 0] >= xlimits[0]) & (occupied[:, 0] <= xlimits[1]) &
            (occupied[:, 1] >= ylimits[0]) & (occupied[:, 1] <= ylimits[1]) &
            (occupied[:, 2] <= 0.02) & (occupied[:, 2] >= 0.01))
        print(contactPoints)
        if contactPoints > 0:
            x_ = np.arange(xlimits[0], xlimits[1], 0.05)
            y_ = np.arange(ylimits[0], ylimits[1], 0.05)
            z_ = np.arange(0, 0.5, 0.05)
            x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
            points = np.array((x.ravel(), y.ravel(), z.ravel())).T
            p2.append(points)
            colors += [index for _ in range(len(points))]
            index += 1

            print(points.shape)

    p2 = np.concatenate(p2)
    p, indices = np.unique(p2, axis=0, return_index=True)
    colors = np.array(colors)
    splot = go.Scatter3d(
        x=p2[indices, 0].flatten(),
        y=p2[indices, 1].flatten(),
        z=p2[indices, 2].flatten(),
        mode='markers',
        marker=dict(color=colors[indices],
                    size=5, opacity=1, colorscale=[
                [0, "rgb(31,120,180)"],
                [0.25, "rgb(178,223,138)"],
                [0.5, "rgb(51,160,44)"],
                [0.75, "rgb(251,154,153)"],
                [1, "rgb(227,26,28)"]])
    )

    lay = go.Layout(autosize=True, scene=dict(camera=dict(eye=dict(x=1.15, y=1.15, z=0.8)), aspectmode='data'))
    fig = go.Figure(data=[splot], layout=lay)
    fig.show()


def getOccupiedVoxels(tree, res):
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

    occupied = np.concatenate(occupied, axis=0)
    if empty:
        empty = np.concatenate(empty, axis=0)

    return occupied, empty


def proximitySensor(tree, poses, rays_for_poses, rng):
    points = []
    for pos, rays in zip(poses, rays_for_poses):
        pts = np.array(rayCast3D(tree, rays, pos, rng))
        pts = np.unique(np.array(pts), axis=0)
        points.append(pts)
    return points


def rayCast3D(tree, rays, pos, rng):
    points = []
    pos = np.array(pos)
    for ray in rays:
        end = np.full((3,), np.nan)
        res = tree.castRay(pos, ray, end, True, rng)
        if res:
            points.append(end)
    return points


def createVisibility3D(tree, rng, converted_poses, rays=None, occupied=None):
    visib = [set() for _ in range(len(converted_poses))]
    for pos, vis in zip(converted_poses, visib):
        if occupied is not None:
            rays = occupied - pos
            rays = rays[np.linalg.norm(rays, axis=1) < rng, :]
            rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        points = np.array(rayCast3D(tree, rays, pos, rng))
        vis |= set(zip(points[:, 0], points[:, 1], points[:, 2]))
    return visib


def createSphere(fov=180, res=2, res2=2, show_fig=False):
    points = []

    for theta in np.deg2rad(np.arange(-180, 180, res)):
        for phi in np.deg2rad(np.arange(0, 180, res2)):
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points.append((x, y, z))

    points = np.unique(np.array(points), axis=0)

    # ceiling_rays = points[points[:,2] <= 0, :]
    # ground_rays = points[points[:,2] >= 0, :]
    # wall1_rays = points[points[:,0] >= 0, :]
    # wall2_rays = points[points[:,1] >= 0, :]
    # wall3_rays = points[points[:,0] <= 0, :]
    # wall4_rays = points[points[:,1] <= 0, :]
    if show_fig:
        # visu_points = [points, ceiling_rays, ground_rays, wall1_rays, wall2_rays, wall3_rays, wall4_rays]
        # for pts in visu_points:
        ax = plt.axes(projection='3d')
        ax.scatter3D(*points.T)
        plt.show()

    return points  # , ceiling_rays, ground_rays, wall1_rays, wall2_rays, wall3_rays, wall4_rays


def padSensorData(poses, occupied):
    points = []
    for xlimits, ylimits in poses:
        contactPoints = np.sum(
            (occupied[:, 0] >= xlimits[0]) & (occupied[:, 0] <= xlimits[1] + RESOLUTION) &
            (occupied[:, 1] >= ylimits[0]) & (occupied[:, 1] <= ylimits[1] + RESOLUTION) &
            (occupied[:, 2] >= 0.01) & (occupied[:, 2] <= 0.02))  # skip ground voxels
        if contactPoints > 0:
            x_ = np.arange(xlimits[0], xlimits[1] + RESOLUTION / 2, RESOLUTION / 2)
            y_ = np.arange(ylimits[0], ylimits[1] + RESOLUTION / 2, RESOLUTION / 2)
            z_ = np.arange(0, 3, RESOLUTION / 2)
            x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
            points.append(np.array((x.ravel(), y.ravel(), z.ravel())).T)

    return points


def gateSensorData(poses, occupied):
    points = []
    for xlimits, ylimits, zlimits in poses:
        contactPoints = np.sum(
            (occupied[:, 0] >= xlimits[0]) & (occupied[:, 0] <= xlimits[1] + RESOLUTION) &
            (occupied[:, 1] >= ylimits[0]) & (occupied[:, 1] <= ylimits[1] + RESOLUTION) &
            (occupied[:, 2] >= zlimits[0]) & (occupied[:, 2] <= zlimits[1] + RESOLUTION))

        if contactPoints > 0:  # TODO: nafouknout kvadr - nejenom pruh ale vetsi kus odpovidajici cloveku
            x_ = np.arange(xlimits[0], xlimits[1] + RESOLUTION / 2, RESOLUTION / 2)
            y_ = np.arange(ylimits[0], ylimits[1] + RESOLUTION / 2, RESOLUTION / 2)
            z_ = np.arange(0, 3, RESOLUTION / 2)
            x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
            points.append(np.array((x.ravel(), y.ravel(), z.ravel())).T)

    return points


def depthCameraSensorData(tree, poses, cam_matrices, occupied, rotations):
    visib = [set() for _ in range(len(poses))]
    for pos, vis, cam_matrix, rotation in zip(poses, visib, cam_matrices, rotations):
        rays = occupied - pos
        rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
        points = []
        pos = np.array(pos)
        proj_mat = cam_matrix @ np.hstack((rotation.as_matrix().T, -rotation.as_matrix().T @ pos[:, np.newaxis]))
        for ray in rays:
            end = np.full((3,), np.nan)
            res = tree.castRay(pos, ray, end, True, LASER_RANGE)
            if res:
                proj = proj_mat @ np.array([end[0], end[1], end[2], 1])
                proj /= proj[-1]
                if (0 <= proj[0] <= 2 * cam_matrix[0, 2] and 0 <= proj[1] <= 2 * cam_matrix[1, 2]):
                    points.append(end)

        points = np.array(points)
        vis |= set(zip(points[:, 0], points[:, 1], points[:, 2]))
    return visib


def mergeSensorData(lidar_data, lidar_poses, depthcam_data, depthcam_poses, pad_data, robot_proximity):
    tree = octomap.OcTree(RESOLUTION)
    for points, pose in zip(lidar_data, lidar_poses):
        tree.insertPointCloud(np.array(list(points)), np.array(pose), lazy_eval=True)

    for points, pose in zip(depthcam_data, depthcam_poses):
        tree.insertPointCloud(np.array(list(points)), np.array(pose), lazy_eval=True)

    tree.updateInnerOccupancy()

    for points in pad_data:
        for p in points:
            # n = tree.search(p)
            # try:
            #     tree.isNodeOccupied(n)
            # except:
            tree.updateNode(p, True, lazy_eval=True)

    for point in robot_proximity:
        tree.updateNode(point, True, lazy_eval=True)

    tree.updateInnerOccupancy()

    saveModel(tree, 'pokus.bt')


def saveModel(tree, filename):
    # remove ceiling to see something
    minx, miny, minz = tree.getMetricMin()
    maxx, maxy, maxz = tree.getMetricMax()
    x_ = np.arange(minx + RESOLUTION / 2, maxx + RESOLUTION / 2, RESOLUTION)
    y_ = np.arange(miny + RESOLUTION / 2, maxy + RESOLUTION / 2, RESOLUTION)
    z_ = np.array([maxz])
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    points = np.array((x.ravel(), y.ravel(), z.ravel())).T
    for p in points:
        tree.deleteNode(p, False)

    tree.writeBinary(f'{filename}'.encode())


def inflateRobot(folder, inflation_value):  # inflation_value - size in meters
    tree = octomap.OcTree(RESOLUTION)
    tree.readBinary(f"models/{folder}/robot.bt".encode())
    res = RESOLUTION / 2
    offset = 0.5  # the robot can be inflated in that empty space outside the model
    mins = tree.getMetricMin() - offset
    dims = ((tree.getMetricSize() + 2 * offset + res) / res).astype(int)

    occupied, empty = getOccupiedVoxels(tree, res)
    coordinates = np.zeros_like(occupied, dtype=int)
    coordinates = np.floor((occupied - mins) / res).astype(int)

    image = np.zeros(dims, dtype=int)
    image[coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]] = 1
    dilated = ndi.binary_dilation(image, ndi.generate_binary_structure(rank=3, connectivity=1),
                                  iterations=int(inflation_value / res))

    dilated = np.logical_xor(image, dilated)
    model = octomap.OcTree(RESOLUTION)
    coords = np.argwhere(dilated == 1) * res + mins  # np.array([[minx,miny,minz]])

    # for p in coords:
    # model.updateNode(p, True, lazy_eval=True)

    # for p in occupied:
    #     model.updateNode(p, False, lazy_eval=True)

    # model.updateInnerOccupancy()

    # model.writeBinary(f'robot_pokus.bt'.encode())

    return coords


def proximityRobot(inflation_value, folder):
    tree = octomap.OcTree(RESOLUTION)
    tree.readBinary(f"models/{folder}/human.bt".encode())
    res = RESOLUTION / 2
    occupied, empty = getOccupiedVoxels(tree, res)
    robot_cover = inflateRobot(folder, inflation_value)
    a = set((tuple(np.round(i, 2)) for i in robot_cover))
    b = set((tuple(np.round(i, 2)) for i in occupied))
    return np.array(list(a.intersection(b)))


if __name__ == "__main__":
    folder = "human0-exp2"
    rays = createSphere(res=1, res2=1)  # not used when "occupied" voxels are sent to createVisibility3D function
    gt_tree = octomap.OcTree(RESOLUTION)
    gt_tree.readBinary((f"models/{folder}/model.bt").encode())
    occupied, empty = getOccupiedVoxels(gt_tree, RESOLUTION)
    robot_prox = proximityRobot(2, folder)

    # LIDAR POSES
    ceiling = (0.5, 0., 2.99)  # , Rotation.from_euler('XYZ', [180, 0, 0], degrees=True)
    ground = (0.5, 0., 0.02)  # , Rotation.from_euler('XYZ', [0, 0, 0], degrees=True)
    wall1 = (2.99, 0, 1.5)  # , Rotation.from_euler('XYZ', [0, -90, 0], degrees=True)
    wall2 = (-2.98, 0, 1.5)  # , Rotation.from_euler('XYZ', [0, 90, 0], degrees=True)
    wall3 = (0, -1.98, 1.5)  # , Rotation.from_euler('XYZ', [-90, 0, 0], degrees=True)
    wall4 = (0, 1.99, 1.5)  # , Rotation.from_euler('XYZ', [90, 0, 0], degrees=True)
    lidar_poses = []  # , ground, wall1, wall2, wall3, wall4]

    # keypoints = np.loadtxt(f"scene5.csv", delimiter=",")
    # plotSkeleton(gt_tree, ceiling, keypoints, LASER_RANGE)

    lidar_data = createVisibility3D(gt_tree, LASER_RANGE, lidar_poses, rays, occupied)

    cam_poses = [wall1]
    cam_matrices = [np.array([[120 / np.tan(45 / 2), 0, 120], [0, 80 / np.tan(45 / 2), 80], [0, 0, 1]])]
    # cam_matrices = [np.array([[520, 0, 320], [0, 520, 240], [0, 0, 1]])]
    cam_rotation = [Rotation.from_euler('XYZ', [0, 0, 0], degrees=True)]
    depthcam_data = depthCameraSensorData(gt_tree, cam_poses, cam_matrices, occupied, cam_rotation)

    # PADS poses - ((x_min, x_max), (y_min, y_max)) 
    pads = [((-0.2, 1), (-1, -0.6)), ((-0.2, 1), (1, 1.5))]
    pad_data = []  # padSensorData(pads, occupied)

    # GATES poses - ((x_min, x_max), (y_min, y_max), (z_min, z_max)) 
    gates = [((-2.8, 2.8), (-0.7, -0.7), (1, 1))]
    gates_data = []  # gateSensorData(gates, occupied)

    mergeSensorData(lidar_data, lidar_poses, depthcam_data, cam_poses, pad_data + gates_data, robot_prox)

    # plotVisibility3D(gt_tree, poses, RNG, rays, occupied)
    # plotPadData(pads, occupied)
