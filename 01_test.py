########################################################################
# import default python-library
########################################################################
import os
import glob
import sys
import ipdb
import pickle
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy as np
# from import
from tqdm import tqdm
from sklearn import metrics
# original lib
import common as com
import keras_model
########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
#######################################################################

########################################################################
# main 01_test.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)

    # make output result directory
    os.makedirs(param["result_directory_tf"], exist_ok=True)

    # load base directory
    dirs = com.select_dirs(param=param, mode=mode)

    # initialize lines in csv for AUC and pAUC
    csv_lines = []

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))
        machine_type = os.path.split(target_dir)[1]

        print("============== MODEL LOAD ==============")
        # set model path
        model_file = "{model}/model_{machine_type}.hdf5".format(model=param["model_directory_tf"],
                                                                machine_type=machine_type)

        # load model file
        if not os.path.exists(model_file):
            com.logger.error("{} model not found ".format(machine_type))
            sys.exit(-1)
        model = keras_model.load_model(model_file)
        model.summary()

        # Load test data
        test_files, test_files_label = com.test_file_list_generator(target_dir, id_name="", mode=True)
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
            result=param["result_directory_tf"],
            machine_type=machine_type,
            id_str="all_id")
        anomaly_score_list = []
        anomaly_score_dict = {}

        #### Start testing
        y_predict = [0. for k in test_files]
        # Loop to get the score
        for idx, test_file in tqdm(enumerate(test_data), total=num_files):
            pred = model.predict(test_file)
            errors = np.mean(np.square(test_file - pred), axis=1)
            y_predict[idx] = np.mean(errors)
            anomaly_score_list.append([os.path.basename(test_files[idx]), y_predict[idx]])
            anomaly_score_dict[os.path.basename(test_files[idx])] = y_predict[idx]

        # save anomaly score for all id
        com.save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
        com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv))

        performance_csv = "{result}/result_{machine_type}_{id_str}.csv".format(
            result=param["result_directory_tf"],
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

        # Write CSV for each ID

        machine_id_list = com.get_machine_id_list_for_test(target_dir)
        start = 0
        for i, id_str in enumerate(machine_id_list):
            # load test file
            files, y_true = com.test_file_list_generator(target_dir, id_str, True)
            # setup anomaly score file path
            anomaly_score_csv_id = "{result}/anomaly_score_{machine_type}_{id_str}.csv".format(
                result=param["result_directory_tf"],
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

            # append AUC and pAUC to lists
            # Calculate auc and p_auc
            auc = metrics.roc_auc_score(y_true, y_pred)
            p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])

            csv_lines.append([id_str.split("_", 1)[1], auc, p_auc])
            performance.append([auc, p_auc])
            com.logger.info("AUC : {}".format(auc))
            com.logger.info("pAUC : {}".format(p_auc))

        # calculate averages for AUCs and pAUCs
        averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
        csv_lines.append(["Average"] + list(averaged_performance))
        csv_lines.append([])


    # Saving the final results
    result_path = "{result}/{file_name}".format(result=param["result_directory_tf"],
                                                file_name=param["result_file_tf"])
    com.logger.info("AUC and pAUC results -> {}".format(result_path))
    com.save_csv(save_file_path=result_path, save_data=csv_lines)