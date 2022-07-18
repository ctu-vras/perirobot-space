import numpy as np
import pandas as pd
import os
import sys

PROCESS_KEYPOINTS = False

if __name__ == "__main__":
    if PROCESS_KEYPOINTS:
        experiment_names = [
            'cam_ceil',
            'cam_west',
            'double_cam',
            'cam_pad',
            'cam_prox',
        ]
        prefix = 'keypoint_'
        dtypes = {
                  "mean": np.dtype('f4'),
                  "max": np.dtype('f4'),
                  "IoU_b": np.dtype('f4'),
                  "IoU_s": np.dtype('f4'),
                  "IoU_c": np.dtype('f4')
                 }
        init_idx = 0

    else:
        try:
            experiment_names = [sys.argv[1]]
        except IndexError:
            experiment_names = [
                'cam_ceil',
                'cam_west',
                'lasergate',
                'lidar_ceil',
                'pad',
                'proximity',
                # 'double_cam',
                # 'cam_pad',
                # 'cam_prox',
                # 'lidar_prox',
            ]
        prefix = ''
        dtypes = {"use_case": 'str', "robot_score": np.dtype('f4'),
                  "robot_unkwn_occupied": np.dtype('f4'), "robot_unkwn_empty": np.dtype('f4'),
                  "robot_tpr_occupied": np.dtype('f4'), "robot_tpr_free": np.dtype('f4'),
                  "robot_fpr_occupied": np.dtype('f4'), "robot_fpr_free": np.dtype('f4'),
                  "human_score": np.dtype('f4'), "human_unkwn_occupied": np.dtype('f4'),
                  "human_unkwn_empty": np.dtype('f4'),
                  "human_tpr_occupied": np.dtype('f4'), "human_tpr_free": np.dtype('f4'),
                  "human_fpr_occupied": np.dtype('f4'), "human_fpr_free": np.dtype('f4')}
        init_idx = 1

    col_names = dtypes.keys()
    keys = [x for x in dtypes.keys()]

    report_df = pd.DataFrame(index=keys[init_idx:])

    for experiment_name in experiment_names:
        experiment_path = os.path.join(f"results", experiment_name)

        df = pd.DataFrame([], columns=col_names)

        for experiment in [x[0] for x in os.walk(experiment_path)][1:]:
            stats_file = os.path.join(experiment, f"{prefix}stats.csv")
            data = pd.read_csv(stats_file, delimiter=",",
                               names=col_names,
                               dtype=dtypes,
                               )
            df = df.append(data, ignore_index=True)
            print(experiment)

        means = df.describe().round(2).iloc[1, :]
        try:
            report_df.insert(len(report_df.columns), experiment_name, means.values)
        except:
            import ipdb; ipdb.set_trace()

        print(df)
        means.to_latex(experiment_path + f'/{prefix}description.tex')

    report_df.to_latex(f'results/{prefix}description.tex')
