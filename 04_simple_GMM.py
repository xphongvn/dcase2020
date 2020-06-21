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
import sys

########################################################################
# Set seed
########################################################################
random.seed(0)
np.random.seed(0)

########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()

# check mode
# "development": mode == True
# "evaluation": mode == False
mode = com.command_line_chk()
if mode is None:
    sys.exit(-1)

########################################################################
# Build data set
########################################################################

# make output directory
os.makedirs(param["model_directory_tf"], exist_ok=True)

# load base_directory list
dirs, add_dirs = com.select_dirs(param=param, mode=mode)
print("Select dir: {}".format(dirs))
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

    if param["feature"]["extra_only"]:
        name = "extra_only"
    elif param["feature"]["extra"]:
        name = "logmel_extra"
    else:
        name = "logmel"

    model_file = param["model_directory"] + "/GMM_{}_{}.model".format(name,machine_type)
    # Check if model is exsiting
    if os.path.exists(model_file):
        print("Found existing model at: {}".format(model_file))
        with open(model_file, "rb") as fp:
            clf = pickle.load(fp)
        print("Finished loading model")

    else:
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

        if param["add_data"]:
            add_files = com.file_list_generator(add_dirs[idx])
            add_data = com.list_to_vector_array(add_files,
                                                msg="generate train_dataset",
                                                n_mels=param["feature"]["n_mels"],
                                                frames=param["feature"]["frames"],
                                                n_fft=param["feature"]["n_fft"],
                                                hop_length=param["feature"]["hop_length"],
                                                power=param["feature"]["power"],
                                                extra_features=param["feature"]["extra"],
                                                extra_only=param["feature"]["extra_only"])
            print("Dimension train data: {}".format(train_data.shape))
            print("Dimension additional data: {}".format(add_data.shape))
            train_data = np.vstack((train_data,add_data))
            del add_data

        # fit a Gaussian Mixture Model with two components
        clf = mixture.GaussianMixture(n_components=10, covariance_type='full', tol=1e-3, max_iter=20,
                                      init_params='kmeans', verbose=2)
        clf.fit(train_data)

    ##########################################################################################
    # Test time
    # Load test data
    test_files, test_files_label = com.test_file_list_generator(target_dir, id_name="", mode=mode)
    test_data = com.list_to_vector_array(test_files,
                                         msg="generate test_dataset",
                                         n_mels=param["feature"]["n_mels"],
                                         frames=param["feature"]["frames"],
                                         n_fft=param["feature"]["n_fft"],
                                         hop_length=param["feature"]["hop_length"],
                                         power=param["feature"]["power"],
                                         extra_features=param["feature"]["extra"],
                                         extra_only=param["feature"]["extra_only"])

    num_files = len(test_files)
    num_features = test_data.shape[1]
    # Number of rows for each data file
    num_rows = int(len(test_data) / num_files)
    test_data = test_data.reshape(num_files, num_rows, num_features)

    # setup anomaly score file path
    anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{id_str}.csv".format(
        result=param["result_directory"],
        machine_type=machine_type,
        id_str="all_id")
    anomaly_score_list = []
    anomaly_score_dict = {}

    #### Start testing
    y_predict = [0. for k in test_files]
    # Loop to get the score
    for idx, test_file in tqdm(enumerate(test_data), total=num_files):
        prob = clf.score_samples(test_file)
        avg_prob = -np.mean(prob)
        y_predict[idx] = avg_prob
        anomaly_score_list.append([os.path.basename(test_files[idx]), y_predict[idx]])
        anomaly_score_dict[os.path.basename(test_files[idx])] = y_predict[idx]

    # save anomaly score for all id
    com.save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
    com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv))

    if mode:
        performance_csv = "{result}/result_{machine_type}_{id_str}.csv".format(
            result=param["result_directory"],
            machine_type=machine_type,
            id_str="all_id")

        # AUC and PAUC for all
        auc = metrics.roc_auc_score(test_files_label, y_predict)
        p_auc = metrics.roc_auc_score(test_files_label, y_predict, max_fpr=param["max_fpr"])

        com.logger.info("AUC for all: {}".format(auc))
        com.logger.info("pAUC for all: {}".format(p_auc))

        performance_all = []
        performance_all.append(["AUC", "p_AUC"])
        performance_all.append([auc, p_auc])

        com.save_csv(save_file_path=performance_csv, save_data=performance_all)
        com.logger.info("Performance result ->  {}".format(performance_csv))

        # results by type
        csv_lines.append([machine_type])
        csv_lines.append(["id", "AUC", "pAUC"])
        performance = []

    #Write CSV for each ID
    machine_id_list = com.get_machine_id_list_for_test(target_dir)
    start = 0
    for i, id_str in enumerate(machine_id_list):
        # load test file
        files, y_true = com.test_file_list_generator(target_dir, id_str, mode)
        # setup anomaly score file path
        anomaly_score_csv_id = "{result}/anomaly_score_{machine_type}_{id_str}.csv".format(
            result=param["result_directory"],
            machine_type=machine_type,
            id_str=id_str)
        # Get anomaly score from anomaly_score_dict
        num = len(files)
        anomaly_score_list_id = []
        y_pred = []
        for file in files:
            y_pred.append(anomaly_score_dict[os.path.basename(file)])
            anomaly_score_list_id.append([os.path.basename(file), y_pred[-1]])
        # save anomaly score for each id
        com.save_csv(save_file_path=anomaly_score_csv_id, save_data=anomaly_score_list_id)
        com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv_id))

        if mode:
            # append AUC and pAUC to lists
            # Calculate auc and p_auc
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])

            csv_lines.append([id_str.split("_", 1)[1], auc, p_auc])
            performance.append([auc, p_auc])
            com.logger.info("AUC : {}".format(auc))
            com.logger.info("pAUC : {}".format(p_auc))

    if mode:
        # calculate averages for AUCs and pAUCs
        averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
        csv_lines.append(["Average"] + list(averaged_performance))
        csv_lines.append([])

    # save model
    if not os.path.exists(model_file):
        print("Saving model to: {}".format(model_file))
        with open(model_file, 'wb') as fp:
            pickle.dump(clf, fp, protocol=4)
        print("Finished saving model")

# Saving the final results
result_path = "{result}/{file_name}".format(result=param["result_directory"],file_name=param["result_file"])
com.logger.info("AUC and pAUC results -> {}".format(result_path))
com.save_csv(save_file_path=result_path, save_data=csv_lines)