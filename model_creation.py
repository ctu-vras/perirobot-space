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


def prepareRobot(joints, robot_tf):
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
    m.transform(robot_tf)
    pcd = m.sample_points_poisson_disk(30000)
    return np.asarray(pcd.points)


def createCell():
    x_ = np.arange(-3, 3.005, 0.005)
    y_ = np.arange(-2, 2.005, 0.005)
    z_ = np.array([0, 3])
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    points = np.array((x.ravel(),y.ravel(), z.ravel())).T
    x_ = np.array([-3, 3])
    y_ = np.arange(-2, 2.005, 0.005)
    z_ = np.arange(0, 3.005, 0.005)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    points1 = np.array((x.ravel(),y.ravel(), z.ravel())).T
    x_ = np.arange(-3, 3.005, 0.005)
    y_ = np.array([-2, 2])
    z_ = np.arange(0,3.005, 0.005)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    points2 = np.array((x.ravel(),y.ravel(), z.ravel())).T
    cell = np.concatenate((points, points1, points2))
    return cell


def createTable():
    x_ = np.arange(-0.2, 1.505, 0.005)
    y_ = np.arange(-0.5, 0.505, 0.005)
    z_ = np.array([0.9,1.0])
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    points = np.array((x.ravel(),y.ravel(), z.ravel())).T
    x_ = np.array([-0.2, 1.5])
    y_ = np.arange(-0.5, 0.505, 0.005)
    z_ = np.arange(0.9,1.005, 0.005)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    points1 = np.array((x.ravel(),y.ravel(), z.ravel())).T
    x_ = np.arange(-0.2, 1.505, 0.005)
    y_ = np.array([-0.5, 0.5])
    z_ = np.arange(0.9,1.005, 0.005)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    points2 = np.array((x.ravel(),y.ravel(), z.ravel())).T

    
    x1_ = np.arange(-0.05, 0.055, 0.005)
    y1_ = np.array([-0.05, 0.05])
    z1_ = np.arange(0, 0.905, 0.005)
    x1, y1, z1 = np.meshgrid(x1_, y1_, z1_, indexing='ij')
    x2_ = np.array([-0.05, 0.05])
    y2_ = np.arange(-0.05, 0.055, 0.005)
    z2_ = np.arange(0, 0.905, 0.005)
    x2, y2, z2 = np.meshgrid(x2_, y2_, z2_, indexing='ij')
    points3 = np.concatenate((np.array((x1.ravel(),y1.ravel(), z1.ravel())).T,np.array((x2.ravel(),y2.ravel(), z2.ravel())).T))
    table = np.concatenate((points, points1, points2, points3+np.array([[-0.15, -0.45 ,0]]), points3+np.array([[1.45, -0.45 ,0]]), points3+np.array([[1.45, 0.45 ,0]]), points3+np.array([[-0.15, 0.45 ,0]])))
    return table



def addHuman(filename, T):

    mesh = o3d.io.read_triangle_mesh(f"data/human/{filename}")
    mesh.compute_vertex_normals()
    mesh.transform(T)
    pcd = mesh.sample_points_poisson_disk(80000)
    #o3d.visualization.draw_geometries([pcd])
    return np.asarray(pcd.points)



if __name__ == "__main__":
    
    joints = [0, 0,-np.pi/2,np.pi/2,-np.pi/2,-np.pi/2,0]
    
    T = np.eye(4) # robot TF
    T[:3,-1] = [0,0,1]
    robot = prepareRobot(joints, T)
    table = createTable()
    cell = createCell()
    filenames = ['operator-v1.stl', '02zman22-v1.stl', 'dummy.stl', 'waiting.stl', 'what.stl', 'man.stl', 'man2.stl']
    Ts = [[0.2,-0.8,0,0,0, np.pi/2], [0.2,-0.8,0,0,0, np.pi/2], [0.2,-0.8,0,0,0,-np.pi/2], [0.3,-0.8,0,0,0,0], [0.3,-0.8,0,0,0,np.pi/2], [0.2,-0.8,0,0,0,np.pi], [0.5,-0.8,0.3,0,0,np.pi]] # human poses
    
    for i in range(7):
        resolution = 0.01
        model = octomap.OcTree(resolution)
        for p in robot:
            model.updateNode(p, True, lazy_eval=True)
        for p in cell:
            model.updateNode(p, True, lazy_eval=True)
        for p in table:
            model.updateNode(p, True, lazy_eval=True)
        points = addHuman(filenames[i], xyz_rpy_to_matrix(Ts[i]))
        for p in points:
            model.updateNode(p, True, lazy_eval=True)

        model.updateInnerOccupancy()
    

        model.writeBinary(f'scene{i}.bt'.encode())