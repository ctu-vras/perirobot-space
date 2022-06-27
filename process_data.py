import numpy as np
import pandas as pd
import os
import sys


if __name__ == "__main__":
    try:
        experiment_names = sys.argv[1]
    except IndexError:
        experiment_names = [
                            'cam_ceil',
                            'cam_east',
                            'lasergate',
                            'lidar_ceil',
                            'pad',
                            'proximity',
                            ]

    dtypes = {"use_case": 'str', "robot_score": np.dtype('f4'),
              "robot_unkwn_occupied": np.dtype('f4'), "robot_unkwn_empty": np.dtype('f4'),
              "robot_tpr_occupied": np.dtype('f4'), "robot_tpr_free": np.dtype('f4'),
              "human_score": np.dtype('f4'), "human_unkwn_occupied": np.dtype('f4'),
              "human_unkwn_empty": np.dtype('f4'),
              "human_tpr_occupied": np.dtype('f4'), "human_tpr_free": np.dtype('f4')}
    col_names = dtypes.keys()
    keys = [x for x in dtypes.keys()]

    report_df = pd.DataFrame(index=keys[1:])

    for experiment_name in experiment_names:
        experiment_path = os.path.join("results", experiment_name)

        df = pd.DataFrame([], columns=col_names)

        for experiment in [x[0] for x in os.walk(experiment_path)][1:]:
            stats_file = os.path.join(experiment, "stats.csv")
            data = pd.read_csv(stats_file, delimiter=",",
                               names=col_names,
                               dtype=dtypes,
                               )
            df = df.append(data, ignore_index=True)

        means = df.describe().round(2).iloc[1, :]
        try:
            report_df.insert(len(report_df.columns), experiment_name, means.values)
        except:
            import ipdb; ipdb.set_trace()

        print(df)
        print(df.describe().iloc[1, :])
        means.to_latex(experiment_path + '/description.tex')

    report_df.to_latex('results/description.tex')
