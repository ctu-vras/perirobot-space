import numpy as np
import octomap
import open3d as o3d
import os

# nose, leye, reye, lear, rear, lshoulder, rshoulder, lelbow, relbow, lwrist, rwrist, lhip, rhip, lknee, rknee, lankle, rankle
kpt_operator = np.array([[0.084623, 0.099352, 1.721721], [0.023685, 0.121650, 1.716806], [0.084623, 0.099352, 1.721721],
                         [-0.047704, 0.060070, 1.730057], [0.102995, -0.024719, 1.730057],
                         [-0.023750, 0.219361, 1.514450],
                         [0.101512, -0.214513, 1.514450], [-0.002789, 0.211081, 1.186939],
                         [0.047996, -0.232904, 1.187355],
                         [0.196541, 0.268050, 1.151146], [0.305339, -0.108851, 1.118090],
                         [0.087791, 0.106545, 1.039094],
                         [0.030806, -0.196894, 0.987761], [0.016430, 0.130227, 0.576269],
                         [0.034519, -0.092172, 0.575352],
                         [-0.107752, 0.182742, 0.116204], [0.001353, -0.133227, 0.130098]])

kpt_02zman = np.array([[0.199843, 0.005302, 1.570330], [0.198014, 0.024809, 1.599651], [0.189723, -0.013900, 1.581241],
                       [0.086792, -0.068368, 1.603126], [0.136901, 0.076595, 1.607546], [0.007240, 0.169528, 1.391199],
                       [0.013305, -0.188459, 1.355872], [0.338953, 0.187511, 1.388029],
                       [-0.245927, -0.239666, 1.201583],
                       [0.540609, 0.002711, 1.387663], [-0.398084, -0.213514, 0.975665],
                       [-0.017205, 0.108629, 0.904159],
                       [-0.011567, -0.103755, 0.980481], [-0.293933, 0.131154, 0.366492],
                       [0.259705, -0.079067, 0.942106],
                       [-0.538413, 0.083971, 0.167305], [0.694411, -0.073306, 0.781873]])

kpt_dummy = np.array(
    [[-0.122258, 0.008754, 1.690146], [-0.119117, -0.035298, 1.743374], [-0.121225, 0.008742, 1.754903],
     [-0.084599, -0.068702, 1.708583], [-0.084599, 0.071269, 1.708555], [-0.001809, -0.225482, 1.499987],
     [0.036627, 0.227763, 1.508449], [0.035315, -0.239328, 1.175673], [0.021082, 0.254620, 1.156755],
     [0.012013, -0.277437, 0.899290], [0.044857, 0.303971, 0.914129], [-0.098424, -0.095983, 1.036513],
     [-0.098737, 0.097902, 1.036506], [-0.045290, -0.132587, 0.510858], [-0.053128, 0.103443, 0.467241],
     [-0.062870, -0.141764, 0.117005], [-0.032980, 0.153388, 0.121586]])

kpt_waiting = np.array([[0.039667, 0.008902, 1.455261], [0.056552, 0.039692, 1.491407], [-0.008906, 0.001625, 1.504609],
                        [0.046848, 0.140242, 1.479336], [-0.071786, 0.113518, 1.471275], [0.165951, 0.153886, 1.382787],
                        [-0.148374, 0.164846, 1.379172], [0.300434, 0.174273, 1.142824],
                        [-0.283716, 0.217087, 1.111333],
                        [0.314457, 0.118032, 0.887518], [-0.266733, 0.181221, 0.888018], [0.120321, 0.025713, 0.928115],
                        [-0.123813, 0.070346, 0.997842], [0.106918, -0.173525, 0.573382],
                        [-0.094995, -0.129336, 0.537074],
                        [-0.057759, -0.322522, 0.118825], [-0.042038, -0.185678, 0.061736]])

kpt_what = np.array([[0.044662, -0.212447, 1.476554], [0.115063, -0.222244, 1.522151], [0.034611, -0.222211, 1.522571],
                     [0.129139, -0.066935, 1.501686], [0.013554, -0.179062, 1.486828], [0.232653, -0.042797, 1.296842],
                     [-0.152577, -0.029305, 1.347926], [0.364290, 0.027239, 1.232991], [-0.240701, 0.037318, 1.240793],
                     [0.592338, 0.060947, 1.074394], [-0.203378, 0.009551, 1.033542], [0.107488, -0.106254, 1.007980],
                     [-0.085805, -0.088326, 0.951682], [0.033032, -0.154152, 0.532943],
                     [-0.074166, -0.052112, 0.508021],
                     [-0.070939, 0.050637, 0.104202], [-0.138896, -0.021474, 0.088471]])

kpt_man = np.array([[-0.007877, -0.070648, 1.712497], [0.031653, -0.058819, 1.801641], [-0.049058, -0.053805, 1.806113],
                    [0.071907, -0.008977, 1.734773], [-0.081114, -0.015231, 1.763429], [0.221464, 0.025710, 1.553174],
                    [-0.232268, 0.037406, 1.553174], [0.231720, -0.034536, 1.229116], [-0.315509, 0.062955, 1.241731],
                    [0.273618, -0.100535, 0.940469], [-0.201534, -0.144110, 1.015252], [0.144696, -0.106344, 1.080000],
                    [-0.122214, -0.115914, 1.080000], [0.114644, -0.103831, 0.509375], [-0.156952, -0.070007, 0.492148],
                    [0.137371, -0.163605, 0.074261], [-0.138516, -0.131723, 0.088789]])

kpt_man2 = np.array(
    [[-0.013541, -0.264595, 1.472349], [0.031556, -0.272287, 1.535822], [-0.045260, -0.270554, 1.538319],
     [0.116072, -0.188921, 1.514339], [-0.127212, -0.158589, 1.505670], [0.273210, -0.050114, 1.320030],
     [-0.278515, -0.062712, 1.299473], [0.280538, -0.268188, 1.087925], [-0.283807, -0.272948, 1.068670],
     [0.304270, -0.476790, 0.914764], [-0.296549, -0.487369, 0.906378], [0.098503, 0.151820, 0.803782],
     [-0.152350, 0.174799, 0.813164], [0.256617, -0.185153, 0.614159], [-0.296445, -0.177177, 0.624600],
     [0.248389, 0.025850, 0.172184], [-0.279160, 0.023638, 0.202715]])

keypoints = [kpt_operator, kpt_02zman, kpt_dummy, kpt_waiting, kpt_what, kpt_man, kpt_man2]


def rpy_to_matrix(coords):
    coords = np.asanyarray(coords)
    c3, c2, c1 = np.cos(coords)
    s3, s2, s1 = np.sin(coords)

    return np.round(np.array([
        [c1 * c2, (c1 * s2 * s3) - (c3 * s1), (s1 * s3) + (c1 * c3 * s2)],
        [c2 * s1, (c1 * c3) + (s1 * s2 * s3), (c3 * s1 * s2) - (c1 * s3)],
        [-s2, c2 * s3, c2 * c3]
    ]), 5)


def xyz_rpy_to_matrix(xyz_rpy):
    matrix = np.eye(4)
    matrix[:3, 3] = xyz_rpy[:3]
    matrix[:3, :3] = rpy_to_matrix(xyz_rpy[3:])
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

    xyz_rpy_values = [[0, 0, 0, 0, 0, 0], [0, 0, d1, 0, 0, 0], [0, s_off, 0, 0, np.pi / 2, 0], [0, e_off, -a2, 0, 0, 0],
                      [0, 0, -a3, 0, np.pi / 2, 0], [0, d4 - e_off - s_off, 0, 0, 0, 0], [0, 0, d5, 0, 0, 0]]
    joint_axes = [0, 2, 1, 1, 1, 2, 1]
    names = ['base.stl', 'shoulder.stl', 'upperarm.stl', 'forearm.stl', 'wrist1.stl', 'wrist2.stl', 'wrist3.stl']
    m = o3d.geometry.TriangleMesh()
    T = np.eye(4)
    for i in range(7):
        mesh = o3d.io.read_triangle_mesh(f"data/{names[i]}")
        mesh.compute_vertex_normals()
        vals = np.copy(xyz_rpy_values[i])
        vals[joint_axes[i] + 3] += joints[i]

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
    points = np.array((x.ravel(), y.ravel(), z.ravel())).T
    x_ = np.array([-3, 3])
    y_ = np.arange(-2, 2.005, 0.005)
    z_ = np.arange(0, 3.005, 0.005)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    points1 = np.array((x.ravel(), y.ravel(), z.ravel())).T
    x_ = np.arange(-3, 3.005, 0.005)
    y_ = np.array([-2, 2])
    z_ = np.arange(0, 3.005, 0.005)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    points2 = np.array((x.ravel(), y.ravel(), z.ravel())).T
    cell = np.concatenate((points, points1, points2))
    return cell


def createTable():
    x_ = np.arange(-0.2, 1.505, 0.005)
    y_ = np.arange(-0.5, 0.505, 0.005)
    z_ = np.array([0.9, 1.0])
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    points = np.array((x.ravel(), y.ravel(), z.ravel())).T
    x_ = np.array([-0.2, 1.5])
    y_ = np.arange(-0.5, 0.505, 0.005)
    z_ = np.arange(0.9, 1.005, 0.005)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    points1 = np.array((x.ravel(), y.ravel(), z.ravel())).T
    x_ = np.arange(-0.2, 1.505, 0.005)
    y_ = np.array([-0.5, 0.5])
    z_ = np.arange(0.9, 1.005, 0.005)
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    points2 = np.array((x.ravel(), y.ravel(), z.ravel())).T

    x1_ = np.arange(-0.05, 0.055, 0.005)
    y1_ = np.array([-0.05, 0.05])
    z1_ = np.arange(0, 0.905, 0.005)
    x1, y1, z1 = np.meshgrid(x1_, y1_, z1_, indexing='ij')
    x2_ = np.array([-0.05, 0.05])
    y2_ = np.arange(-0.05, 0.055, 0.005)
    z2_ = np.arange(0, 0.905, 0.005)
    x2, y2, z2 = np.meshgrid(x2_, y2_, z2_, indexing='ij')
    points3 = np.concatenate(
        (np.array((x1.ravel(), y1.ravel(), z1.ravel())).T, np.array((x2.ravel(), y2.ravel(), z2.ravel())).T))
    table = np.concatenate((points, points1, points2, points3 + np.array([[-0.15, -0.45, 0]]),
                            points3 + np.array([[1.45, -0.45, 0]]), points3 + np.array([[1.45, 0.45, 0]]),
                            points3 + np.array([[-0.15, 0.45, 0]])))
    return table


def addHuman(filename, kpts, T):
    mesh = o3d.io.read_triangle_mesh(f"data/human/{filename}")
    mesh.compute_vertex_normals()
    kpts_h = np.hstack((kpts, np.ones((kpts.shape[0], 1))))
    keypoints = T @ kpts_h.T
    mesh.transform(T)
    pcd = mesh.sample_points_poisson_disk(80000)
    # o3d.visualization.draw_geometries([pcd])
    return np.asarray(pcd.points), np.round(keypoints[:3, :].T, 2)


def composeModel(res, points_array, name):
    tree = octomap.OcTree(res)
    for points in points_array:
        for p in points:
            tree.updateNode(p, True, lazy_eval=True)
    tree.updateInnerOccupancy()
    tree.writeBinary(f'models/{name}/model.bt'.encode())


if __name__ == "__main__":

    joints = [0, 0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0]

    T = np.eye(4)  # robot TF
    T[:3, -1] = [0, 0, 1]
    robot = prepareRobot(joints, T)
    table = createTable()
    cell = createCell()
    filenames = ['operator-v1.stl', '02zman22-v1.stl', 'dummy.stl', 'waiting.stl', 'what.stl', 'man.stl', 'man2.stl']
    Ts = [[0.2, -0.8, 0, 0, 0, np.pi / 2], [0.2, -0.8, 0, 0, 0, np.pi / 2], [0.2, -0.8, 0, 0, 0, -np.pi / 2],
          [0.3, -0.8, 0, 0, 0, 0], [0.3, -0.8, 0, 0, 0, np.pi / 2], [0.2, -0.8, 0, 0, 0, np.pi],
          [0.5, -0.8, 0.3, 0, 0, np.pi]]  # human poses

    resolution = 0.05  # resolution for 5 cm -> 2 cm
    robot_model = octomap.OcTree(resolution)
    for p in robot:
        robot_model.updateNode(p, True, lazy_eval=True)
    robot_model.updateInnerOccupancy()

    for i in range(7):
        os.makedirs(f'models/human{i}-exp2', exist_ok=True)
        human_points, kpts = addHuman(filenames[i], keypoints[i], xyz_rpy_to_matrix(Ts[i]))
        human_model = octomap.OcTree(resolution)
        for p in human_points:
            human_model.updateNode(p, True, lazy_eval=True)
        human_model.updateInnerOccupancy()
        human_model.writeBinary(f'models/human{i}-exp2/human.bt'.encode())
        composeModel(resolution, [robot, cell, table, human_points], f"human{i}-exp2")
        np.savetxt(f"models/human{i}-exp2/keypoints.csv", kpts, fmt='%1.2f', delimiter=",")
        np.savez(f"models/human{i}-exp2/vars.npz", joints, T, Ts[i])
        robot_model.writeBinary(f'models/human{i}-exp2/robot.bt'.encode())

    i = 6
    human_points, kpts = addHuman(filenames[i], keypoints[i], xyz_rpy_to_matrix(Ts[i]))
    j = 3
    human_points2, kpts2 = addHuman(filenames[j], keypoints[j], xyz_rpy_to_matrix([0.3, 0.8, 0, 0, 0, np.pi]))
    os.makedirs(f'models/human{i}{j}-exp2', exist_ok=True)
    human_model = octomap.OcTree(resolution)
    for p in human_points:
        human_model.updateNode(p, True, lazy_eval=True)
    human_model.updateInnerOccupancy()
    human_model.writeBinary(f'models/human{i}{j}-exp2/human.bt'.encode())
    composeModel(resolution, [robot, cell, table, human_points, human_points2], f"human{i}{j}-exp2")
    np.savetxt(f"models/human{i}{j}-exp2/keypoints.csv", kpts, fmt='%1.2f', delimiter=",")
    np.savez(f"models/human{i}{j}-exp2/vars.npz", joints, T, Ts[i])
    robot_model.writeBinary(f'models/human{i}{j}-exp2/robot.bt'.encode())

    os.makedirs(f'models/nohuman-exp2', exist_ok=True)
    composeModel(resolution, [robot, cell, table], "nohuman-exp2")
    robot_model.writeBinary(f'models/nohuman-exp2/robot.bt'.encode())
