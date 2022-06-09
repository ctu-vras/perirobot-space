import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import octomap


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


def getOccupiedVoxels(tree, res):
    occupied = []
    for it in tree.begin_leafs():
        center = it.getCoordinate()
        dimension = max(1, round(it.getSize() / res))
        origin = center - (dimension / 2 - 0.5) * res
        indices = np.column_stack(np.nonzero(np.ones((dimension, dimension, dimension))))
        points = origin + indices * res

        if tree.isNodeOccupied(it):
            occupied.append(points)

    return  np.concatenate(occupied, axis=0)



def getOccupiedAndEmptyVoxels(tree, res):
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
    print(occupied.shape)
    return occupied, empty



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


def createSphere(fov=150, res=2, res2=2):
    points = []

    for theta in np.deg2rad(np.arange(-180, 180, res)):
        for phi in np.deg2rad(np.arange(0, fov + 1, res2)):
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points.append((x, y, z))

    return np.array(points)


if __name__ == "__main__":
    rays = createSphere(res=5, res2=5)
    resolution = 0.1
    gt_tree = octomap.OcTree(resolution)
    gt_tree.readBinary(("pokus.bt").encode())
    RNG = 10
    occupied = getOccupiedVoxels(gt_tree, resolution)

    poses = [(2.,2.,1.)]
    orig_visibility = createVisibility3D(gt_tree, RNG, poses, rays, occupied)
    

    plotVisibility3D(gt_tree, poses, RNG, rays, occupied)