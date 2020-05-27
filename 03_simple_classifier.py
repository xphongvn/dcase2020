#### Simple Classifier for Anomaly vs Normal

import torch
import torch.nn as nn
from torch_model import BinaryClassifier
import common as com
import os
import numpy as np
import ipdb
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
import time
import torch_utils as tu
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

########################################################################
# Set seed
########################################################################
com.deterministic_everything(100, pytorch=True, tf=False)

########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()

########################################################################
# Build data set
########################################################################

# load base_directory list
dirs = com.select_dirs(param=param, mode=True)

# loop of the base directory
for idx, target_dir in enumerate(dirs):
    print("\n===========================")
    print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx + 1, total=len(dirs)))

    # Get machine types and train_data
    machine_type = os.path.split(target_dir)[1]
    machine_id_list = com.get_machine_id_list_for_test(target_dir)
    for idn, id_str in enumerate(machine_id_list):
        files, labels = com.test_file_list_generator(target_dir,id_str, mode=True, dir_name="test")
        train_data = com.list_to_vector_array(files,
                                              msg="generate train_dataset",
                                              n_mels=param["feature"]["n_mels"],
                                              frames=param["feature"]["frames"],
                                              n_fft=param["feature"]["n_fft"],
                                              hop_length=param["feature"]["hop_length"],
                                              power=param["feature"]["power"],
                                              extra_features=param["feature"]["extra"])
        # Get labels into train_data
        n_row = int(train_data.shape[0]/len(labels))
        train_labels = []
        for i in range(len(labels)):
            train_labels.extend([labels[i] for k in range(n_row)]) # Duplicate label into n_row times
        train_labels = np.array(train_labels)

        # first iteration
        if idn == 0:
            all_data = train_data
            all_labels = np.array(train_labels)
        else: # from next iteration, stack
            all_data = np.vstack((all_data,train_data))
            all_labels = np.hstack((all_labels,np.array(train_labels)))

    # Done extract data for with normal and abnormal data for 1 machine type
    model = BinaryClassifier(inputDim=train_data.shape[1]).to(device)
    print(model)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-8)
    X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.5, shuffle=True)
    train_dataset = TensorDataset(Tensor(X_train), Tensor(y_train))
    test_dataset = TensorDataset(Tensor(X_test), Tensor(y_test))
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=param["fit"]["batch_size"],
                              shuffle=param["fit"]["shuffle"],
                              num_workers=2
                              )

    # Pytorch Start training
    train_losses = []
    val_losses = []
    for e in range(param["fit"]["epochs"]):
        time_start = time.time()
        loss_train_epoch = tu.train_epoch(model, optimizer, criterion, train_loader, device)
        train_losses.append(loss_train_epoch)
        loss_eval_epoch = tu.evaluate_acc_epoch(model, criterion, test_dataset, device)
        val_losses.append(loss_eval_epoch)
        time_end = time.time()
        print("Train Epoch: {} [Train loss: {}] [Eval loss: {}] [Time: {}]".format(e,loss_train_epoch,
                                                                                   loss_eval_epoch,
                                                                                   time_end - time_start))

