import pandas as pd

baseline_file = "./result_tf/result_tf.csv"
check_file = "./result_torch/result_torch.csv"

def read_file(file_name):
    all_df = {}
    all_average = {}
    with open(file_name) as f:
        for i in range(5):
            name = f.readline().strip()
            column_name_string = f.readline()
            column_names = column_name_string.strip().split(",")
            check = True
            result={}
            for cl_name in column_names:
                result[cl_name] = []

            while check:
                line = f.readline()
                line_value = line.strip().split(",")
                if line_value[0] == "Average":
                    average_auc = float(line_value[1])
                    average_pauc = float(line_value[2])
                elif line=="\n":
                    break
                else:
                    for i, cl_name in enumerate(column_names):
                        result[cl_name].append(float(line_value[i]))

            all_df[name] = pd.DataFrame(result)
            all_average[name] = {"average_auc":average_auc, "average_pauc":average_pauc}
    return all_df, all_average

def compare_average(baseline_average, check_average):
    for name in baseline_average:
        print(name)
        print("AUC difference with baseline: {}".
              format(check_average[name]["average_auc"] - baseline_average[name]["average_auc"]))
        print("pAUC difference with baseline: {}".
              format(check_average[name]["average_pauc"] - baseline_average[name]["average_pauc"]))

################## MAIN PROGRAM ###################
baseline_dfs, baseline_avgs = read_file(baseline_file)
check_dfs, check_avgs = read_file(check_file)

compare_average(baseline_avgs, check_avgs)
