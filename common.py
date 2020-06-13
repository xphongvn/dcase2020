"""
 @file   common.py
 @brief  Commonly used script
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import python-library
########################################################################
# default
import glob
import argparse
import sys
import os
import re
import os

# additional
import numpy
import librosa
import librosa.core
import librosa.feature
import yaml
import ipdb
from pyAudioAnalysis import audioFeatureExtraction
from tqdm import tqdm
import csv
import itertools
import pickle
import time

########################################################################


########################################################################
# setup STD I/O
########################################################################
"""
Standard output is logged in "baseline.log".
"""
import logging

logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


########################################################################


########################################################################
# version
########################################################################
__versions__ = "1.0.0"
########################################################################


########################################################################
# argparse
########################################################################
def command_line_chk():
    parser = argparse.ArgumentParser(description='Without option argument, it will not run properly.')
    parser.add_argument('-v', '--version', action='store_true', help="show application version")
    parser.add_argument('-e', '--eval', action='store_true', help="run mode Evaluation")
    parser.add_argument('-d', '--dev', action='store_true', help="run mode Development")
    args = parser.parse_args()
    if args.version:
        print("===============================")
        print("DCASE 2020 task 2 baseline\nversion {}".format(__versions__))
        print("===============================\n")
    if args.eval ^ args.dev:
        if args.dev:
            flag = True
        else:
            flag = False
    else:
        flag = None
        print("incorrect argument")
        print("please set option argument '--dev' or '--eval'")
    return flag
########################################################################


########################################################################
# load parameter.yaml
########################################################################
def yaml_load():
    with open("baseline.yaml") as stream:
        param = yaml.safe_load(stream)
    return param

########################################################################


########################################################################
# file I/O
########################################################################
# wav file Input
def file_load(wav_name, mono=False):
    """
    load .wav file.

    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data

    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


########################################################################


########################################################################
# feature extractor
########################################################################
def file_to_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0,
                         extra_features=False,
                         extra_only=False):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    if extra_only:
        return file_to_vector_array_extra_only(file_name, frames, hop_length)

    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa
    y, sr = file_load(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                     sr=sr,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * numpy.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vector_array_size < 1:
        return numpy.empty((0, dims))

    ## Extract extra features if wanted
    # 05.1 extract extra features with pyAudioAnalysis
    if extra_features:
        num_windows = log_mel_spectrogram.shape[1]
        extra_features = extract_new_features(y,sr,num_windows)

        # 05.2 Concatenate extra_features with log_mel_spectrogram
        new_features = numpy.concatenate((extra_features, log_mel_spectrogram), axis=0)
        size = new_features.shape[0] # update dimension with new features added
        dims = size * frames # update dimension with new features added
    else:
        new_features = log_mel_spectrogram # No change
        size = new_features.shape[0]

    # 06 generate feature vectors by concatenating multiframes
    vector_array = numpy.zeros((vector_array_size, dims))

    for t in range(frames):
        vector_array[:, size * t: size * (t + 1)] = new_features[:, t: t + vector_array_size].T

    return vector_array

# Add new features from pyAudioAnalysis
def extract_new_features(data, sr, n):
    # Divided data into n-windows
    data_list = []
    window_size = int(len(data) / n)
    for i in range(n):
        data_list.append(data[window_size*i:window_size*(i+1)])
    data_list = numpy.array(data_list)

    # Extract short-term features
    features, f_names = audioFeatureExtraction.stFeatureExtraction(data, sr, window_size, window_size)

    # Extract statistical features: MEAN, MAX, MIN, VAR
    mean = numpy.mean(data_list, axis=1)
    max = numpy.max(data_list, axis=1)
    min = numpy.min(data_list, axis=1)
    var = numpy.var(data_list, axis=1)

    # concatenate all features
    new_features = numpy.vstack((mean,max,min,var,features))

    return new_features


def file_to_vector_array_extra_only(file_name,
                         frames=5,
                         hop_length=512):
    """
    convert file_name to a vector array with extra features only.

    file_name : str
        target .wav file

    return : numpy.array( numpy.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # # 02 generate melspectrogram using librosa
    y, sr = file_load(file_name)
    vector_array_size = int(len(y)/hop_length) - frames + 1
    num_windows = int(len(y) / hop_length) + 1
    extra_features = extract_new_features(y,sr,num_windows)

    # 05.2 Concatenate extra_features with log_mel_spectrogram
    #new_features = numpy.concatenate((extra_features, log_mel_spectrogram), axis=0)
    new_features = extra_features
    size = new_features.shape[0] # update dimension with new features added
    dims = size * frames # update dimension with new features added

    # 06 generate feature vectors by concatenating multiframes
    vector_array = numpy.zeros((vector_array_size, dims))

    for t in range(frames):
        vector_array[:, size * t: size * (t + 1)] = new_features[:, t: t + vector_array_size].T

    return vector_array

# load dataset
def select_dirs(param, mode):
    """
    param : dict
        baseline.yaml data

    return :
        if active type the development :
            dirs :  list [ str ]
                load base directory list of dev_data
        if active type the evaluation :
            dirs : list [ str ]
                load base directory list of eval_data
    """
    if mode:
        logger.info("load_directory <- development")
        dir_path = os.path.abspath("{base}/*".format(base=param["dev_directory"]))
        dirs = sorted(glob.glob(dir_path))
    else:
        logger.info("load_directory <- evaluation")
        dir_path = os.path.abspath("{base}/*".format(base=param["eval_directory"]))
        dirs = sorted(glob.glob(dir_path))
    return dirs

########################################################################

def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0,
                         extra_features=False,
                         extra_only=False):
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
    if extra_only:
        extra_features = True
        file_name = hash(str(str(file_list) + str(extra_features)) + str(power) + str(hop_length) +
                         str(n_fft) + str(frames) + str(n_mels) + str(extra_only))
    else:
        file_name = hash(str(str(file_list) + str(extra_features)) + str(power) + str(hop_length) +
                         str(n_fft) + str(frames) + str(n_mels))
    load = os.path.exists("./pickle_data/{}.pickle".format(file_name))

    if not load:
        print("Does not find existing extracted pickle features at {}.pickle. Extracting...".format(file_name))
        time_start = time.time()
        # iterate file_to_vector_array()
        if extra_only:
            dataset = list_to_vector_array_extra_only(file_list, msg, n_mels, frames, n_fft, hop_length, power)
        else:
            for idx in tqdm(range(len(file_list)), desc=msg):
                vector_array = file_to_vector_array(file_list[idx],
                                                        n_mels=n_mels,
                                                        frames=frames,
                                                        n_fft=n_fft,
                                                        hop_length=hop_length,
                                                        power=power,
                                                        extra_features=extra_features)
                if idx == 0:
                    dataset = numpy.zeros((vector_array.shape[0] * len(file_list), vector_array.shape[1]), float)
                dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array

            print("Finished extracting features. Time taken: {}s".format(time.time() - time_start))

        # Save pickle
        print("Saving pickle file of the data")
        time_start = time.time()
        os.makedirs("./pickle_data", exist_ok=True)
        with open("./pickle_data/{}.pickle".format(file_name), 'wb') as fp:
            pickle.dump(dataset, fp, protocol=4)
        print("Finished saving pickle file. Time taken: {}s".format(time.time()-time_start))
    else:
        # Load pikcle
        print("Found existing extracted pickle features at: ./pickle_data/{}.pickle".format(file_name))
        time_start = time.time()
        with open("./pickle_data/{}.pickle".format(file_name), 'rb') as fp:
            dataset = pickle.load(fp)
        print("Finished loading pickle file. Time taken: {}s".format(time.time() - time_start))

    return dataset

def list_to_vector_array_extra_only(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0,
                         extra_features=False):

    # iterate file_to_vector_array()
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = file_to_vector_array_extra_only(file_list[idx],
                                                frames=frames,
                                                hop_length=hop_length)
        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list), vector_array.shape[1]), float)
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
    logger.info("target_dir : {}".format(target_dir))

    # generate training list
    training_list_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    files = sorted(glob.glob(training_list_path))
    if len(files) == 0:
        logger.exception("no_wav_file!!")

    logger.info("train_file num : {num}".format(num=len(files)))
    return files


########################################################################
# def
########################################################################
def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


def get_machine_id_list_for_test(target_dir,
                                 dir_name="test",
                                 ext="wav"):
    """
    target_dir : str
        base directory path of "dev_data" or "eval_data"
    test_dir_name : str (default="test")
        directory containing test data
    ext : str (default="wav)
        file extension of audio files

    return :
        machine_id_list : list [ str ]
            list of machine IDs extracted from the names of test files
    """
    # create test files
    dir_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(dir_path))
    # extract id
    machine_id_list = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return machine_id_list


def test_file_list_generator(target_dir,
                             id_name,
                             mode,
                             dir_name="test",
                             prefix_normal="normal",
                             prefix_anomaly="anomaly",
                             ext="wav"):
    """
    target_dir : str
        base directory path of the dev_data or eval_data
    id_name : str
        id of wav file in <<test_dir_name>> directory
    dir_name : str (default="test")
        directory containing test data
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files

    return :
        if the mode is "development":
            test_files : list [ str ]
                file list for test
            test_labels : list [ boolean ]
                label info. list for test
                * normal/anomaly = 0/1
        if the mode is "evaluation":
            test_files : list [ str ]
                file list for test
    """
    logger.info("target_dir : {}".format(target_dir+"_"+id_name))

    # development
    if mode:
        normal_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                 dir_name=dir_name,
                                                                                 prefix_normal=prefix_normal,
                                                                                 id_name=id_name,
                                                                                 ext=ext)))
        normal_labels = numpy.zeros(len(normal_files))
        anomaly_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                  dir_name=dir_name,
                                                                                  prefix_anomaly=prefix_anomaly,
                                                                                  id_name=id_name,
                                                                                  ext=ext)))
        anomaly_labels = numpy.ones(len(anomaly_files))
        files = numpy.concatenate((normal_files, anomaly_files), axis=0)
        labels = numpy.concatenate((normal_labels, anomaly_labels), axis=0)
        logger.info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no_wav_file!!")
        print("\n========================================")

    # evaluation
    else:
        files = sorted(
            glob.glob("{dir}/{dir_name}/*{id_name}*.{ext}".format(dir=target_dir,
                                                                  dir_name=dir_name,
                                                                  id_name=id_name,
                                                                  ext=ext)))
        labels = None
        logger.info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            logger.exception("no_wav_file!!")
        print("\n=========================================")

    return files, labels
########################################################################

# Set seed all
def deterministic_everything(seed=42, pytorch=True, tf=False):
    """Set pseudo random everything deterministic. a.k.a. `seed_everything`
    Universal to major frameworks.
    Thanks to https://docs.fast.ai/dev/test.html#getting-reproducible-results
    Thanks to https://pytorch.org/docs/stable/notes/randomness.html
    """
    import random
    # Python RNG
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Numpy RNG
    import numpy as np
    np.random.seed(seed)

    # Pytorch RNGs
    if pytorch:
        import torch
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # TensorFlow RNG
    if tf:
        import tensorflow as tf
        tf.set_random_seed(seed)