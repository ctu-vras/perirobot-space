import numpy as np
import pers

from scipy.spatial.transform import Rotation


def run_exp1(res):
    all_cam_rots = []
    init_rot = Rotation.from_euler('xyz', [0, 90, 0], degrees=True)
    all_poses = []
    counts = []
    for y_ in [-2 + 2 * res, 2 - res]:
        x_ = np.arange(-2.2, 2.4, 0.5)
        z_ = np.arange(1.5, 2.6, 0.5)
        x, y, z = np.meshgrid(x_, np.array([y_]), z_, indexing='ij')
        all_poses.append(np.array((x.ravel(), y.ravel(), z.ravel())).T)
        counts.append(x.ravel().shape[0])
    for x_ in [-3 + 2 * res, 3 - res]:
        y_ = np.arange(-1.5, 1.6, 0.5)
        z_ = np.arange(1.5, 2.6, 0.5)
        x, y, z = np.meshgrid(np.array([x_]), y_, z_, indexing='ij')
        all_poses.append(np.array((x.ravel(), y.ravel(), z.ravel())).T)
        counts.append(x.ravel().shape[0])
    x_ = np.arange(-2.2, 2.4, 0.5)
    y_ = np.arange(-1.5, 1.6, 0.5)
    z_ = np.array([3 - res])
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    all_poses.append(np.array((x.ravel(), y.ravel(), z.ravel())).T)
    counts.append(x.ravel().shape[0])
    all_poses = np.concatenate(all_poses)

    all_cam_rots.append([Rotation.from_euler('xyz', [0, 0, 90], degrees=True) * init_rot for _ in range(counts[0])])
    all_cam_rots.append([Rotation.from_euler('xyz', [0, 0, -90], degrees=True) * init_rot for _ in range(counts[1])])
    all_cam_rots.append([Rotation.from_euler('xyz', [0, 0, 0], degrees=True) * init_rot for _ in range(counts[2])])
    all_cam_rots.append([Rotation.from_euler('xyz', [0, 0, 180], degrees=True) * init_rot for _ in range(counts[3])])
    all_cam_rots.append([Rotation.from_euler('xyz', [0, 90, 0], degrees=True) * init_rot for _ in range(counts[4])])
    all_cam_rots = np.concatenate(all_cam_rots)

    for idx, (poses, cam_rotations) in enumerate(zip(all_poses, all_cam_rots)):
        for i in range(27):
            case = "human0-dyn-exp3"
            output_name = "exp1/" + case + "_" + str(idx) + "_" + str(i)
            simulation = pers.Pers(case, resolution=res, output_name=output_name, lidar_poses=[poses],
                                   lidar_range=20, noise_lidar=0.01, lidar_mindist=0.5, cam_rotations=[cam_rotations],
                                   add_robot_vol=False, dynamic_model=i)

            simulation.process_sensors(compute_statistics=True, detect_keypoints=False, save_stats=True)


def run_exp2(res):
    cases = ["human1-exp3",
             "human4-exp3",
             "human63-exp3",
             ]

    cam_matrices = [np.array(
        [[360 / np.tan(np.deg2rad(58 / 2)), 0, 360], [0, 640 / np.tan(np.deg2rad(87 / 2)), 640], [0, 0, 1]])]

    all_cam_rots = []
    all_poses = []
    counts = []
    init_rot = Rotation.from_euler('xyz', [0, 90, 0], degrees=True)
    for y_ in [-2 + 2 * res, 2 - res]:
        x_ = np.arange(-2.2, 2.4, 0.5)
        z_ = np.arange(1.5, 2.6, 0.5)
        x, y, z = np.meshgrid(x_, np.array([y_]), z_, indexing='ij')
        all_poses.append(np.array((x.ravel(), y.ravel(), z.ravel())).T)
        counts.append(x.ravel().shape[0])
    for x_ in [-3 + 2 * res, 3 - res]:
        y_ = np.arange(-1.5, 1.6, 0.5)
        z_ = np.arange(1.5, 2.6, 0.5)
        x, y, z = np.meshgrid(np.array([x_]), y_, z_, indexing='ij')
        all_poses.append(np.array((x.ravel(), y.ravel(), z.ravel())).T)
        counts.append(x.ravel().shape[0])
    x_ = np.arange(-2.2, 2.4, 0.5)
    y_ = np.arange(-1.5, 1.6, 0.5)
    z_ = np.array([3 - res])
    x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')
    all_poses.append(np.array((x.ravel(), y.ravel(), z.ravel())).T)
    counts.append(x.ravel().shape[0])
    all_poses = np.concatenate(all_poses)
    all_poses = np.vstack((all_poses, all_poses, all_poses, all_poses, all_poses))

    all_cam_rots.append([Rotation.from_euler('xyz', [0, 0, 90], degrees=True) * init_rot for _ in range(counts[0])])
    all_cam_rots.append([Rotation.from_euler('xyz', [0, 0, -90], degrees=True) * init_rot for _ in range(counts[1])])
    all_cam_rots.append([Rotation.from_euler('xyz', [0, 0, 0], degrees=True) * init_rot for _ in range(counts[2])])
    all_cam_rots.append([Rotation.from_euler('xyz', [0, 0, 180], degrees=True) * init_rot for _ in range(counts[3])])
    all_cam_rots.append([Rotation.from_euler('xyz', [0, 90, 0], degrees=True) * init_rot for _ in range(counts[4])])

    for x in [-20, 20]:
        for y in [-20, 20]:
            all_cam_rots.append(
                [Rotation.from_euler('xyz', [0, x, 90 + y], degrees=True) * init_rot for _ in range(counts[0])])
            all_cam_rots.append(
                [Rotation.from_euler('xyz', [0, x, -90 + y], degrees=True) * init_rot for _ in range(counts[1])])
            all_cam_rots.append(
                [Rotation.from_euler('xyz', [0, x, y], degrees=True) * init_rot for _ in range(counts[2])])
            all_cam_rots.append(
                [Rotation.from_euler('xyz', [0, x, 180 + y], degrees=True) * init_rot for _ in range(counts[3])])
            all_cam_rots.append(
                [Rotation.from_euler('xyz', [0, 90 + x, y], degrees=True) * init_rot for _ in range(counts[4])])

    all_cam_rots = np.concatenate(all_cam_rots)

    for idx, (poses, cam_rotations) in enumerate(zip(all_poses, all_cam_rots)):
        for case in cases:
            output_name = "exp2/" + case + "_r" + str(idx)
            simulation = pers.Pers(case, resolution=res, output_name=output_name, rgbd_poses=[poses],
                                   cam_matrices=cam_matrices, cam_rotations=[cam_rotations],
                                   rgbd_range=6, noise_rgbd=0.06, rgbd_mindist=0.6, add_robot_vol=False)

            simulation.process_sensors(compute_statistics=True, detect_keypoints=True, save_stats=True)

    for idx, (cam_poses, cam_rotations) in enumerate(zip(all_poses, all_cam_rots)):
        for case in cases:
            output_name = "exp2/" + case + "_c" + str(idx)
            simulation = pers.Pers(case, resolution=res, output_name=output_name, cam_matrices=cam_matrices,
                                   cam_rotations=[cam_rotations], cam_poses=[cam_poses],
                                   robot_zone=[[[-0.5, 2], [-1, 1], [0, 3]]], add_robot_vol=False)

            simulation.process_sensors(compute_statistics=True, detect_keypoints=False, save_stats=True)


def run_exp3(res):
    cases = ["human1-exp3",
             "human4-exp3",
             "human63-exp3",
             ]
    init_rot = Rotation.from_euler('xyz', [0, 90, 0], degrees=True)
    all_pad_poses = []
    rgbd_poses = 9 * [[0.1, 0, 1.15]]
    all_cam_rots = [Rotation.from_euler('xyz', [0, 0, -10 * x], degrees=True) * init_rot for x in range(-4, 5)]
    cam_matrices = [np.array(
        [[360 / np.tan(np.deg2rad(58 / 2)), 0, 360], [0, 640 / np.tan(np.deg2rad(87 / 2)), 640], [0, 0, 1]])]

    for y in [(-1.5, -0.75), (-1.25, -0.5), (0.5, 1.25), (0.75, 1.5)]:
        for x in [(-1.5, -0.5), (-1, 0), (-0.5, 0.5), (0, 1), (0.5, 1.5), (1, 2)]:
            all_pad_poses.append([x, y])

    for case in cases:
        for idx, pad_poses in enumerate(all_pad_poses):
            for j, (rgbd_pose, cam_rotations) in enumerate(zip(rgbd_poses, all_cam_rots)):
                for robot_prox in [0.1, 0.2, 0.3]:
                    output_name = "exp3/" + "p" + str(idx).zfill(2) + "_c" + str(j) + "_r" + str(int(10 * robot_prox))
                    simulation = pers.Pers(case, resolution=res, output_name=output_name, rgbd_poses=[rgbd_pose],
                                           cam_matrices=cam_matrices, cam_rotations=[cam_rotations], rgbd_range=3,
                                           noise_rgbd=0.04, rgbd_mindist=0.3, pad_poses=[pad_poses],
                                           robot_inflation_value=robot_prox, noise_rob_prox=0.01, add_robot_vol=True)

                    simulation.process_sensors(compute_statistics=True, detect_keypoints=False, save_stats=True)

        for idx, pad_poses in enumerate(all_pad_poses):
            for robot_prox in [0.1, 0.2, 0.3]:
                output_name = "exp3/" + "p" + str(idx).zfill(2) + "_r" + str(int(10 * robot_prox))
                simulation = pers.Pers(case, resolution=res, output_name=output_name,
                                       pad_poses=[pad_poses], robot_inflation_value=robot_prox,
                                       noise_rob_prox=0.01, add_robot_vol=True)

                simulation.process_sensors(compute_statistics=True, detect_keypoints=False, save_stats=True)

        for idx, pad_poses in enumerate(all_pad_poses):
            for j, (rgbd_pose, cam_rotations) in enumerate(zip(rgbd_poses, all_cam_rots)):
                output_name = "exp3/" + "p" + str(idx).zfill(2) + "_c" + str(j)
                simulation = pers.Pers(case, resolution=res, output_name=output_name,
                                       rgbd_poses=[rgbd_pose], cam_matrices=cam_matrices,
                                       cam_rotations=[cam_rotations], rgbd_range=3, noise_rgbd=0.04, rgbd_mindist=0.3,
                                       pad_poses=[pad_poses], add_robot_vol=True)

                simulation.process_sensors(compute_statistics=True, detect_keypoints=False, save_stats=True)

        for j, (rgbd_pose, cam_rotations) in enumerate(zip(rgbd_poses, all_cam_rots)):
            for robot_prox in [0.1, 0.2, 0.3]:
                output_name = "exp3/" + "c" + str(j) + "_r" + str(int(10 * robot_prox))
                simulation = pers.Pers(case, resolution=res, output_name=output_name,
                                       rgbd_poses=[rgbd_pose], cam_matrices=cam_matrices,
                                       cam_rotations=[cam_rotations], rgbd_range=3, noise_rgbd=0.04, rgbd_mindist=0.3,
                                       robot_inflation_value=robot_prox, noise_rob_prox=0.01, add_robot_vol=True)

                simulation.process_sensors(compute_statistics=True, detect_keypoints=False, save_stats=True)

        for idx, pad_poses in enumerate(all_pad_poses):
            output_name = "exp3/" + "p" + str(idx).zfill(2)
            simulation = pers.Pers(case, resolution=res, output_name=output_name,
                                   pad_poses=[pad_poses], add_robot_vol=True)

            simulation.process_sensors(compute_statistics=True, detect_keypoints=False, save_stats=True)

        for robot_prox in [0.1, 0.2, 0.3]:
            output_name = "exp3/" "r" + str(int(10 * robot_prox))
            simulation = pers.Pers(case, resolution=res, output_name=output_name,
                                   robot_inflation_value=robot_prox, noise_rob_prox=0.01, add_robot_vol=True)

            simulation.process_sensors(compute_statistics=True, detect_keypoints=False, save_stats=True)

        for j, (rgbd_pose, cam_rotations) in enumerate(zip(rgbd_poses, all_cam_rots)):
            output_name = "exp3/" + "c" + str(j)
            simulation = pers.Pers(case, resolution=res, output_name=output_name, rgbd_poses=[rgbd_pose],
                                   cam_matrices=cam_matrices, cam_rotations=[cam_rotations], rgbd_range=3,
                                   noise_rgbd=0.04, rgbd_mindist=0.3, add_robot_vol=True)

            simulation.process_sensors(compute_statistics=True, detect_keypoints=False, save_stats=True)


if __name__ == "__main__":
    resolution = 0.05
    # run_exp1(resolution)
    # run_exp2(resolution)
    run_exp3(resolution)
