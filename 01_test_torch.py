########################################################################
# import default python-library
########################################################################
import os
import glob
import csv
import re
import itertools
import sys
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
# from import
from tqdm import tqdm
from sklearn import metrics
# original lib
import common as com
import torch_model
import torch
from torch.autograd import Variable
import ipdb
########################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################
# main 01_test_torch.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)

    # make output result directory
    os.makedirs(param["result_directory_torch"], exist_ok=True)

    # load base directory
    dirs = com.select_dirs(param=param, mode=mode)

    # initialize lines in csv for AUC and pAUC
    csv_lines = []

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))
        machine_type = os.path.split(target_dir)[1]
        machine_id_list = com.get_machine_id_list_for_test(target_dir)

        print("============== MODEL LOAD ==============")
        # set model path
        model_file = "{model}/model_{machine_type}.torch".format(model=param["model_directory_torch"],
                                                                machine_type=machine_type)

        # load model file
        if not os.path.exists(model_file):
            com.logger.error("{} model not found ".format(machine_type))
            sys.exit(-1)

        # Load one sample to get the inputDim
        test_files, y_true = com.test_file_list_generator(target_dir, machine_id_list[0], mode)

        data = com.file_to_vector_array(test_files[0],
                                        n_mels=param["feature"]["n_mels"],
                                        frames=param["feature"]["frames"],
                                        n_fft=param["feature"]["n_fft"],
                                        hop_length=param["feature"]["hop_length"],
                                        power=param["feature"]["power"],
                                        extra_features=param["feature"]["extra"])

        input_dim = data.shape[1]
        print("Loading model file: {}".format(model_file))
        model = torch_model.load_model(model_file, input_dim)
        print(model)

        if mode:
            # results by type
            csv_lines.append([machine_type])
            csv_lines.append(["id", "AUC", "pAUC"])
            performance = []



        for id_str in machine_id_list:
            # load test file
            test_files, y_true = com.test_file_list_generator(target_dir, id_str, mode)

            # setup anomaly score file path
            anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{id_str}.csv".format(
                                                                                     result=param["result_directory_torch"],
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

                    feed_data = torch.as_tensor(data, device=device, dtype=torch.float32)
                    with torch.no_grad():
                        pred_cpu = model(feed_data).to('cpu').detach().numpy().copy()
                    errors = numpy.mean(numpy.square(data - pred_cpu), axis=1)
                    y_pred[file_idx] = numpy.mean(errors)
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                except IOError:
                    com.logger.error("file broken!!: {}".format(file_path))

            # save anomaly score
            com.save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
            com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv))

            if mode:
                # append AUC and pAUC to lists
                auc = metrics.roc_auc_score(y_true, y_pred)
                p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])
                csv_lines.append([id_str.split("_", 1)[1], auc, p_auc])
                performance.append([auc, p_auc])
                com.logger.info("AUC : {}".format(auc))
                com.logger.info("pAUC : {}".format(p_auc))

            print("\n============ END OF TEST FOR A MACHINE ID ============")

        if mode:
            # calculate averages for AUCs and pAUCs
            averaged_performance = numpy.mean(numpy.array(performance, dtype=float), axis=0)
            csv_lines.append(["Average"] + list(averaged_performance))
            csv_lines.append([])

    if mode:
        # output results
        result_path = "{result}/{file_name}".format(result=param["result_directory_torch"], file_name=param["result_file_torch"])
        com.logger.info("AUC and pAUC results -> {}".format(result_path))
        com.save_csv(save_file_path=result_path, save_data=csv_lines)
