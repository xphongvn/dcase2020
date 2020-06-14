import pandas as pd
import matplotlib.pylab as plt

baseline_file = "./result/result_tf/result_tf.csv"
#check_file = "./result/result_tf_unet/result_tf_unet.csv"
#check_file = "./result/GMM_extra_only/result_tf.csv"
#check_file = "./result/result_tf_with_extra/result_tf.csv"
check_file = "./result/result_tf_extra_only/result.csv"
#check_file = "./result/result_torch/result_torch.csv"

def read_file(file_name):
    all_df = {}
    all_average = {}
    with open(file_name) as f:
        for i in range(6):
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
    total_dif_auc = 0
    total_dif_pauc = 0
    for name in baseline_average:
        print(name)
        dif_auc = check_average[name]["average_auc"] - baseline_average[name]["average_auc"]
        print("AUC difference with baseline: {}".
              format(dif_auc))
        total_dif_auc += dif_auc
        dif_pauc = check_average[name]["average_pauc"] - baseline_average[name]["average_pauc"]
        print("pAUC difference with baseline: {}".
              format(dif_pauc))
        total_dif_pauc += dif_pauc
        total_dif_pauc += total_dif_pauc

    print("Total difference in AUC: {}".format(total_dif_auc))
    print("Total difference in pAUC: {}".format(total_dif_pauc))

################## MAIN PROGRAM ###################
baseline_dfs, baseline_avgs = read_file(baseline_file)
check_dfs, check_avgs = read_file(check_file)

compare_average(baseline_avgs, check_avgs)

baseline_auc = [item["average_auc"] for item in baseline_avgs.values()]
check_auc = [item["average_auc"] for item in check_avgs.values()]
label = [item for item in baseline_avgs.keys()]
df = {}
for i, value in enumerate(label):
    df[value] = [baseline_auc[i], check_auc[i]]

df = pd.DataFrame(df, index=["baseline","our_model"])
#%%

df.T.plot.bar()
plt.show()