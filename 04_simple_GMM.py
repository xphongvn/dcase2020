#### Simple Classifier for Anomaly vs Normal
import common as com
import os
import numpy as np
import ipdb
import time
from sklearn import mixture, metrics
import random
from tqdm import tqdm

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

# loop of the base directory
for idx, target_dir in enumerate(dirs):
    print("\n===========================")
    print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx + 1, total=len(dirs)))

    # set path
    machine_type = os.path.split(target_dir)[1]

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
                                          extra_features=param["feature"]["extra"])

    # fit a Gaussian Mixture Model with two components
    clf = mixture.GaussianMixture(n_components=3, covariance_type='diag', tol=1e-6, max_iter=500, init_params='kmeans')
    clf.fit(train_data)
    ##########################################################################################
    # Test time
    machine_id_list = com.get_machine_id_list_for_test(target_dir)

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
        for file_idx, file_path in tqdm(enumerate(test_files), total=len(test_files)):
            try:
                data = com.file_to_vector_array(file_path,
                                                n_mels=param["feature"]["n_mels"],
                                                frames=param["feature"]["frames"],
                                                n_fft=param["feature"]["n_fft"],
                                                hop_length=param["feature"]["hop_length"],
                                                power=param["feature"]["power"],
                                                extra_features=param["feature"]["extra"])
                prob = clf.score_samples(data)
                avg_prob = -np.mean(prob)
                y_pred[file_idx] = avg_prob
                anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])

            except IOError:
                com.logger.error("file broken!!: {}".format(file_path))

        # Calculate auc and p_auc
        auc = metrics.roc_auc_score(y_true, y_pred)
        p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])

        print("AUC : {}".format(auc))
        print("P_AUC : {}".format(p_auc))