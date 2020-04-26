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
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
# from import
from tqdm import tqdm
# original lib
import common as com
import keras_model
# Pytorch
# import torch_model
# from torch.utils.data.sampler import SubsetRandomSampler

########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################


########################################################################
# visualizer
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib
        matplotlib.use('PS') #to fix Mac OS
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(30, 10))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save png file path.

        return : None
        """
        self.plt.savefig(name)


########################################################################


def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        vector array for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    # calculate the number of dimensions
    dims = n_mels * frames

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = com.file_to_vector_array(file_list[idx],
                                                n_mels=n_mels,
                                                frames=frames,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                power=power)
        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), dims), float)
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

    return dataset


def file_list_generator(target_dir,
                        dir_name="train",
                        ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    dir_name : str (default="train")
        directory name containing training data
    ext : str (default="wav")
        file extension of audio files

    return :
        train_files : list [ str ]
            file list for training
    """
    com.logger.info("target_dir : {}".format(target_dir))

    # generate training list
    training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    files = sorted(glob.glob(training_list_path))
    if len(files) == 0:
        com.logger.exception("no_wav_file!!")

    com.logger.info("train_file num : {num}".format(num=len(files)))
    return files
########################################################################


########################################################################
# main 00_train.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)
        
    # make output directory
    os.makedirs(param["model_directory"], exist_ok=True)

    # initialize the visualizer
    visualizer = visualizer()

    # load base_directory list
    dirs = com.select_dirs(param=param, mode=mode)

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))

        # set path
        machine_type = os.path.split(target_dir)[1]
        model_file_path = "{model}/model_{machine_type}.hdf5".format(model=param["model_directory"],
                                                                     machine_type=machine_type)
        history_img = "{model}/history_{machine_type}.png".format(model=param["model_directory"],
                                                                  machine_type=machine_type)

        if os.path.exists(model_file_path):
            com.logger.info("model exists")
            continue

        # generate dataset
        print("============== DATASET_GENERATOR ==============")
        files = file_list_generator(target_dir)
        train_data = list_to_vector_array(files,
                                          msg="generate train_dataset",
                                          n_mels=param["feature"]["n_mels"],
                                          frames=param["feature"]["frames"],
                                          n_fft=param["feature"]["n_fft"],
                                          hop_length=param["feature"]["hop_length"],
                                          power=param["feature"]["power"])

        # train model
        print("============== MODEL TRAINING ==============")
        model = keras_model.get_model(param["feature"]["n_mels"] * param["feature"]["frames"])

        # Pytorch
        # model = torch_model.get_model(param["feature"]["n_mels"] * param["feature"]["frames"])

        model.summary()
        # Pytorch
        # print(model)

        model.compile(**param["fit"]["compile"])
        # Pytorch
        # criterion = nn.MSELoss()
        # optimizer = torch.optim.Adam(model.parameters())

        # Behavior of keras: Split training and validation with the defined percentage, and then shuffle
        # the training; validation is not shuffled
        # https://keras.io/getting-started/faq/#how-is-the-validation-split-computed
        history = model.fit(train_data,
                            train_data,
                            epochs=param["fit"]["epochs"],
                            batch_size=param["fit"]["batch_size"],
                            shuffle=param["fit"]["shuffle"],
                            validation_split=param["fit"]["validation_split"],
                            verbose=param["fit"]["verbose"])
        
        #PyTorch
        """
        
        # class GetDatasetDraft(Dataset):
        #     def __init__(self, data, transform=None):
        #         self.data = data
        #         self.target = data #Default
        #         self.transform = transform
        # 
        #     def __getitem__(self, index):
        #         x = self.data[index]
        #         y = self.target[index]
        #         return x, y
        # 
        #     def __len__(self):
        #         return len(self.data)

                
        # split = DataSplit(dataset, shuffle=param["fit"]["shuffle], validation_split=param["fit"]["validation_split"])
        # train_loader, val_loader = split.get_split(batch_size=param["fit"]["batch_size"], num_workers=2)
        
        # trainloader = torch.utils.data.DataLoader(train_data, 
        #                                         batch_size=param["fit"]["batch_size"],
        #                                         shuffle=param["fit"]["shuffle"],
        #                                         num_workers=2)
        
        
        
        class GetDataset(Dataset):

            def __init__(self, data, validation_split=param["fit"]["validation_split"], isValidation=False):
                super(GetDataset, self).__init__()
                numpy_pred = data
                numpy_real = data
                
                if (validation_split):
                    split = int(data.shape[0] * (1. - validation_split))
                
                    if(isValidation):
                        numpy_pred = numpy_pred[split:]
                        numpy_real = numpy_real[split:]
                    else:
                        numpy_pred = numpy_pred[:split]
                        numpy_real = numpy_real[:split]
                
                self.X = numpy_pred
                self.y = numpy_real
        
            def __getitem__(self, index):
                return (self.X[index], self.y[index])
        
            def __len__(self):
                return self.X.shape[0]

        train_dataset = GetDataset(isValidation=False)
        val_dataset = GetDataset(isValidation=True)
                        
        train_loader = DataLoader(dataset=train_dataset, 
                                  batch_size=param["fit"]["batch_size"],
                                  shuffle=param["fit"]["shuffle"],
                                  num_workers=2
                                  )
        val_loader = DataLoader(dataset=val_dataset, 
                                  batch_size=param["fit"]["batch_size"],
                                  shuffle=False,
                                  num_workers=2
                                  )


        for e in range(param["fit"]["epochs"]):
            epoch_loss = 0.0
            for i, (feature, target) in enumerate(dataloader):
                feature, target = feature.cuda(), target.cuda()
                optimizer.zero_grad()
                #forward
                output = model(feature)
                loss = criterion(output, target)
                #backward
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                # ===================log========================
                print('epoch [{}/{}], loss:{:.4f}'
                    .format(epoch + 1, num_epochs, loss.data[0]))

        for e in range(param["fit"]["epochs"]):
            epoch_loss = 0.0
            


            """

        visualizer.loss_plot(history.history["loss"], history.history["val_loss"])
        visualizer.save_figure(history_img)
        model.save(model_file_path)




        com.logger.info("save_model -> {}".format(model_file_path))
        print("============== END TRAINING ==============")
