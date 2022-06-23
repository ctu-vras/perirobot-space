import pandas as pd
import os
import sys


if __name__ == "__main__":
    experiment_name = sys.argv[1]
    experiment_path = os.path.join("results", experiment_name)

    col_names = ["use_case", "robot_score", "robot_unkwn_occupied", "robot_unkwn_empty",
                 "human_score", "human_unkwn_occupied", "human_unkwn_empty"]
    df = pd.DataFrame([], columns=col_names)

    for experiment in [x[0] for x in os.walk(experiment_path)][1:]:
        stats_file = os.path.join(experiment, "stats.csv")
        data = pd.read_csv(stats_file, delimiter=",", names=col_names)
        df = df.append(data, ignore_index=True)

    print(df)
    print(df.describe())
    df.describe().to_latex(experiment_path + '/description.latex')
