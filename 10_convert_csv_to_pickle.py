import pickle
import glob
import os

def read_csv(file_name):
    scores = []
    files = []
    with open(file_name) as f:
        while True:
            line = f.readline().strip().split(",")
            if len(line)==2:
                files.append(line[0])
                scores.append(float(line[1]))
            else:
                break
    return files, scores

def read_folder(dir):
    file_names = sorted(glob.glob("{dir}/anomaly_score_*_id_*.csv".format(dir=dir)))
    return file_names

def get_label(file):
    file = file.split("_")
    if file[0] == "normal":
        return 0
    elif file[0] == "anomaly":
        return 1
    else:
        return None


####################### MAIN PROGRAM #######################
folder_name = "./result/GMM_extra_only_eval"
save_pickle = folder_name + "/" + "GMM_34_features_eval.pickle"

file_names = read_folder(folder_name)

result_dict = {}
for file_name in file_names:
    files, scores = read_csv(file_name)
    file_name = os.path.basename(file_name)
    file_name = file_name.split(".")[0]
    file_name = file_name.split("_")
    machine_name = file_name[2]
    if machine_name not in result_dict:
        result_dict[machine_name] = {}
    if file_name[0] == "id":
        id_str = file_name[0] + file_name[1]
    elif file_name[3] == "id":
        id_str = file_name[3] + "_" + file_name[4]
    else:
        raise("Cannot find id_str")
    id_dict = {}
    for i, file in enumerate(files):
        true_label = get_label(file)
        id_dict[file] = [true_label, scores[i]]
    result_dict[machine_name][id_str] = id_dict

print(result_dict)

with open(save_pickle, "wb") as fp:
    pickle.dump(result_dict, fp)

for machine in result_dict:
    print("Machine {} has {} id".format(machine, len(result_dict[machine])))
    for id in result_dict[machine]:
        print("\tID {} has {} files".format(id, len(result_dict[machine][id])))