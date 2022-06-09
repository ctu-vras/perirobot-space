import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import octomap
import open3d as o3d


def rpy_to_matrix(coords):
    coords = np.asanyarray(coords)
    c3, c2, c1 = np.cos(coords)
    s3, s2, s1 = np.sin(coords)

    return np.round(np.array([
        [c1 * c2, (c1 * s2 * s3) - (c3 * s1), (s1 * s3) + (c1 * c3 * s2)],
        [c2 * s1, (c1 * c3) + (s1 * s2 * s3), (c3 * s1 * s2) - (c1 * s3)],
        [-s2, c2 * s3, c2 * c3]
    ]),5)


def xyz_rpy_to_matrix(xyz_rpy):
    matrix = np.eye(4)
    matrix[:3,3] = xyz_rpy[:3]
    matrix[:3,:3] = rpy_to_matrix(xyz_rpy[3:])
    return matrix


def prepareRobot(joints):
    d1 = 0.1273
    a2 = -0.612
    a3 = -0.5723
    d4 = 0.163941
    d5 = 0.1157
    d6 = 0.0922
    s_off = 0.220941
    e_off = -0.1719

    xyz_rpy_values = [[0,0,0,0,0,0], [0,0, d1, 0,0,0], [0,s_off,0, 0, np.pi/2, 0], [0, e_off, -a2, 0,0,0], [0,0,-a3, 0, np.pi/2, 0], [0,d4-e_off-s_off,0,0,0,0], [0,0,d5,0,0,0]]
    pcd = o3d.geometry.PointCloud()
    Ts = [np.eye(4), np.eye(4), np.eye(4), np.eye(4), np.eye(4), np.eye(4), np.eye(4)]
    joint_axes = [0,2,1,1,1,2,1]
    names = ['base.stl', 'shoulder.stl', 'upperarm.stl', 'forearm.stl', 'wrist1.stl', 'wrist2.stl', 'wrist3.stl']
    points = [5000, 15000, 15000, 15000, 12000, 12000, 5000]
    m = o3d.geometry.TriangleMesh()
    T = np.eye(4)
    for i in range(7):
        mesh = o3d.io.read_triangle_mesh(f"data/{names[i]}")
        mesh.compute_vertex_normals()
        vals = np.copy(xyz_rpy_values[i])
        vals[joint_axes[i]+3] += joints[i]
        
        T = T @ xyz_rpy_to_matrix(vals)
        mesh.transform(T)
        m += mesh
    
    pcd = m.sample_points_poisson_disk(50000)
    o3d.visualization.draw_geometries([pcd])
    return np.asarray(pcd.points)


if __name__ == "__main__":
    resolution = 0.01
    model = octomap.OcTree(resolution)
    joints = [0, 0,-np.pi/2,np.pi/2,-np.pi/2,-np.pi/2,0]
    
    points = prepareRobot(joints)
    for p in points:
        model.updateNode(p, True)
    model.writeBinary(b'robot.bt')