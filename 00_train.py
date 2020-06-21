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
import sys
import numpy as np
########################################################################


########################################################################
# import additional python-library
########################################################################
# original lib
import common as com
import keras_model
import ipdb
from visualizer import visualizer
from keras.callbacks import EarlyStopping

########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
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
    os.makedirs(param["model_directory_tf"], exist_ok=True)

    # initialize the visualizer
    visualizer = visualizer()

    # load base_directory list
    dirs, add_dirs = com.select_dirs(param=param, mode=mode)

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))

        # set path
        machine_type = os.path.split(target_dir)[1]
        model_file_path = "{model}/model_{machine_type}.hdf5".format(model=param["model_directory_tf"],
                                                                     machine_type=machine_type)
        history_img = "{model}/history_{machine_type}.png".format(model=param["model_directory_tf"],
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
                                              power=param["feature"]["power"],
                                              extra_features=param["feature"]["extra"],
                                              extra_only=param["feature"]["extra_only"])

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

        #num_file = len(files)
        #train_data = train_data.reshape(num_file, int(train_data.shape[0] / num_file), train_data.shape[1])

        # train model
        print("============== MODEL TRAINING ==============")
        model = keras_model.get_model(train_data.shape[1])

        model.summary()

        model.compile(**param["fit"]["compile"])

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

        # Behavior of keras: Split training and validation with the defined percentage, and then shuffle
        # the training; validation is not shuffled
        # https://keras.io/getting-started/faq/#how-is-the-validation-split-computed
        history = model.fit(train_data,
                            train_data,
                            epochs=param["fit"]["epochs"],
                            batch_size=param["fit"]["batch_size"],
                            shuffle=param["fit"]["shuffle"],
                            validation_split=param["fit"]["validation_split"],
                            verbose=param["fit"]["verbose"],
                            callbacks=[es])

        visualizer.loss_plot(history.history["loss"], history.history["val_loss"])
        visualizer.save_figure(history_img)
        model.save(model_file_path)


        com.logger.info("save_model -> {}".format(model_file_path))
        print("============== END TRAINING ==============")
