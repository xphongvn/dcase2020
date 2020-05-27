"""
 @file   00_train.py
 @brief  Script for training
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import default python-library
########################################################################
import os
import glob
import sys
import time
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
# from import
from tqdm import tqdm
# original lib
import common as com
from visualizer import visualizer
from torch_utils import GetDataset
import torch_utils as tu


#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch_model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################
# main 00_train_torch.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)

    # make output directory
    os.makedirs(param["model_directory_torch"], exist_ok=True)

    # initialize the visualizer
    visualizer = visualizer()

    # load base_directory list
    dirs = com.select_dirs(param=param, mode=mode)

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx + 1, total=len(dirs)))

        # set path
        machine_type = os.path.split(target_dir)[1]
        model_file_path = "{model}/model_{machine_type}.torch".format(model=param["model_directory_torch"],
                                                                     machine_type=machine_type)
        history_img = "{model}/history_{machine_type}.png".format(model=param["model_directory_torch"],
                                                                  machine_type=machine_type)

        if os.path.exists(model_file_path):
            com.logger.info("model exists")
            continue

        # generate dataset
        print("============== DATASET_GENERATOR ==============")
        files = com.file_list_generator(target_dir)
        train_data = com.list_to_vector_array(files,
                                          msg="generate train_dataset",
                                          n_mels=param["feature"]["n_mels"],
                                          frames=param["feature"]["frames"],
                                          n_fft=param["feature"]["n_fft"],
                                          hop_length=param["feature"]["hop_length"],
                                          power=param["feature"]["power"])

        # train model
        print("============== MODEL TRAINING ==============")

        # Pytorch load model with input dimension
        model = torch_model.get_model(train_data.shape[1])
        print(model)

        # Pytorch's model compile with loss and optimizer
        if param["fit"]["compile"]["loss"] == "mean_squared_error":
            criterion = nn.MSELoss()
        else:
            raise("Not implemented other loss function!")

        if param["fit"]["compile"]["optimizer"] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        else:
            raise("Not implemented other optimizer!")

        # PyTorch's loading dataset for training and validating

        train_dataset = GetDataset(data=train_data, isValidation=False)
        val_dataset = GetDataset(data=train_data, isValidation=True)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=param["fit"]["batch_size"],
                                  shuffle=param["fit"]["shuffle"],
                                  num_workers=2
                                  )
        # Batch loader
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=param["fit"]["batch_size"],
                                shuffle=False,
                                num_workers=2
                                )

        # Pytorch Start training
        train_losses = []
        val_losses = []
        for e in range(param["fit"]["epochs"]):
            time_start = time.time()
            loss_train_epoch = tu.train_epoch(model, optimizer, criterion, train_loader, device)
            train_losses.append(loss_train_epoch)
            loss_eval_epoch = tu.evaluate_epoch(model, criterion, val_loader, device)
            val_losses.append(loss_eval_epoch)
            time_end = time.time()
            print("Train Epoch: {} [Train loss: {}] [Eval loss: {}] [Time: {}]".format(e, loss_train_epoch,
                                                                                       loss_eval_epoch,
                                                                                       time_end-time_start))

        visualizer.loss_plot(train_losses, val_losses)
        visualizer.save_figure(history_img)

        # Pytorch save model
        torch.save(model.state_dict(), model_file_path)

        com.logger.info("save_model -> {}".format(model_file_path))
        print("============== END TRAINING ==============")
