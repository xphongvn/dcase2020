#### Simple Classifier for Anomaly vs Normal
import common as com
import os
import numpy as np
import ipdb
import time
from sklearn import mixture, metrics
import random
from tqdm import tqdm
import pickle

########################################################################
# Set seed
########################################################################
random.seed(0)
np.random.seed(0)

########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()

########################################################################
# Build data set
########################################################################

# make output directory
os.makedirs(param["model_directory_tf"], exist_ok=True)

# load base_directory list
dirs = com.select_dirs(param=param, mode=True)

# initialize lines in csv for AUC and pAUC
csv_lines = []

# make output result directory
os.makedirs(param["result_directory"], exist_ok=True)
os.makedirs(param["model_directory"], exist_ok=True)

# loop of the base directory
for idx, target_dir in enumerate(dirs):
    print("\n===========================")
    print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx + 1, total=len(dirs)))

    # set path
    machine_type = os.path.split(target_dir)[1]

    # Check if model is exsiting

    # generate dataset
    print("============== DATASET_GENERATOR ==============")
    files = com.file_list_generator(target_dir)
    train_data = com.list_to_vector_array(files,
                                          msg="generate train_dataset",
                                          n_mels=param["feature"]["n_mels"],
                                          frames=param["feature"]["frames"],
                                          n_fft=param["feature"]["n_fft"],
                                          hop_length=param["feature"]["hop_length"],
                                          power=param["feature"]["power"],
                                          extra_features=param["feature"]["extra"],
                                          extra_only=param["feature"]["extra"])

    # fit a Gaussian Mixture Model with two components
    clf = mixture.GaussianMixture(n_components=10, covariance_type='full', tol=1e-6, max_iter=100,
                                  init_params='kmeans', verbose=2)
    clf.fit(train_data)
    ##########################################################################################
    # Test time
    machine_id_list = com.get_machine_id_list_for_test(target_dir)

    # results by type
    csv_lines.append([machine_type])
    csv_lines.append(["id", "AUC", "pAUC"])
    performance = []

    for id_str in machine_id_list:
        # load test file
        test_files, y_true = com.test_file_list_generator(target_dir, id_str, True)

        # setup anomaly score file path
        anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{id_str}.csv".format(
            result=param["result_directory_tf"],
            machine_type=machine_type,
            id_str=id_str)
        anomaly_score_list = []

        print("\n============== BEGIN TEST FOR A MACHINE ID ==============")
        y_pred = [0. for k in test_files]
        for file_idx, file_path in enumerate(test_files):
            data = com.file_to_vector_array(file_path,
                                            n_mels=param["feature"]["n_mels"],
                                            frames=param["feature"]["frames"],
                                            n_fft=param["feature"]["n_fft"],
                                            hop_length=param["feature"]["hop_length"],
                                            power=param["feature"]["power"],
                                            extra_features=param["feature"]["extra"],
                                            extra_only=param["feature"]["extra"])
            prob = clf.score_samples(data)
            avg_prob = -np.mean(prob)
            y_pred[file_idx] = avg_prob
            anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])

        # save anomaly score
        com.save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
        com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv))


        # append AUC and pAUC to lists
        # Calculate auc and p_auc
        auc = metrics.roc_auc_score(y_true, y_pred)
        p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])

        print("AUC : {}".format(auc))
        print("P_AUC : {}".format(p_auc))

        csv_lines.append([id_str.split("_", 1)[1], auc, p_auc])
        performance.append([auc, p_auc])
        com.logger.info("AUC : {}".format(auc))
        com.logger.info("pAUC : {}".format(p_auc))

        print("\n============ END OF TEST FOR A MACHINE ID ============")


    # calculate averages for AUCs and pAUCs
    averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
    csv_lines.append(["Average"] + list(averaged_performance))
    csv_lines.append([])

    # output results
    result_path = "{result}/{file_name}_{id_str}".format(result=param["result_directory"],
                                            file_name=param["result_file"], id_str=id_str)
    com.logger.info("AUC and pAUC results -> {}".format(result_path))
    com.save_csv(save_file_path=result_path, save_data=csv_lines)

    anomaly_score_csv = "{result}/anomaly_score_{machine_type}.csv".format(result=param["result_directory"],
                                                                           machine_type=machine_type)
    com.save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
    com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv))

    if param["feature"]["extra_only"]:
        name = "extra_only"
    elif param["feature"]["extra"]:
        name = "logmel_extra"
    else:
        name = "logmel"
    #save model
    with open(param["model_directory"] + "/GMM_{}_{}.model".format(name,machine_type), 'wb') as fp:
        pickle.dump(clf, fp, protocol=4)