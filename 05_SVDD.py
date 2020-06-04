import torch
import logging
import random
import numpy as np
import common as com
import os
import ipdb
from sklearn import metrics
from tqdm import tqdm
from SVDD.datasets.dcase import DCASE_Dataset
from SVDD.utils.config import Config
from SVDD.deepSVDD import DeepSVDD

########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()


# Configuration hard coded
# TODO: move to config file
load_config = None
net_name = "dcase"
xp_path = param["result_directory_SVDD"]
data_path = ""
load_con2fig = None
load_model = None
objective = "one-class"
nu = 0.1
device = "cuda"
seed = 0
optimizer_name = "adam"
lr = 0.0001
n_epochs = 10
lr_milestone = ([5])
batch_size = 512
weight_decay = 0.5e-6
pretrain = True
ae_optimizer_name = "adam"
ae_lr = 0.001
ae_n_epochs = 10
ae_lr_milestone = ([5])
ae_batch_size = 512
ae_weight_decay = 0.5e-3
n_jobs_dataloader = 2
normal_class = 0

if __name__ == '__main__':
    """
        Deep SVDD, a fully deep method for anomaly detection.

        :arg DATASET_NAME: Name of the dataset to load.
        :arg NET_NAME: Name of the neural network to use.
        :arg XP_PATH: Export path for logging the experiment.
        :arg DATA_PATH: Root path of data.
        """

    # Set up logging
    log_file = xp_path + '/log.txt'
    logging.basicConfig(level=logging.DEBUG, filename=log_file)
    logger = logging.getLogger()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Get configuration
    cfg = Config(locals().copy())

    # Print arguments
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)

    logger.info('Normal class: %d' % normal_class)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print configuration
    logger.info('Deep SVDD objective: %s' % cfg.settings['objective'])
    logger.info('Nu-paramerter: %.2f' % cfg.settings['nu'])

    # Set seed
    if cfg.settings['seed'] != -1:
        random.seed(cfg.settings['seed'])
        np.random.seed(cfg.settings['seed'])
        torch.manual_seed(cfg.settings['seed'])
        logger.info('Set seed to %d.' % cfg.settings['seed'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # load base_directory list
    dirs = com.select_dirs(param=param, mode=True)

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        # set path
        machine_type = os.path.split(target_dir)[1]

        # Load train data
        files = com.file_list_generator(target_dir)
        train_data = com.list_to_vector_array(files,
                                             msg="generate train_dataset",
                                             n_mels=param["feature"]["n_mels"],
                                             frames=param["feature"]["frames"],
                                             n_fft=param["feature"]["n_fft"],
                                             hop_length=param["feature"]["hop_length"],
                                             power=param["feature"]["power"],
                                             extra_features=param["feature"]["extra"])
        # Get labels into train_data
        train_labels = np.full(train_data.shape[0], 0)
        if train_data.shape[0] != len(train_labels):
            raise("Train data and labels do not have the same size")

        # Load test data
        test_files, test_files_label = com.test_file_list_generator(target_dir, id_name="", mode=True)
        test_data = com.list_to_vector_array(test_files,
                                             msg="generate test_dataset",
                                             n_mels=param["feature"]["n_mels"],
                                             frames=param["feature"]["frames"],
                                             n_fft=param["feature"]["n_fft"],
                                             hop_length=param["feature"]["hop_length"],
                                             power=param["feature"]["power"],
                                             extra_features=param["feature"]["extra"])

        # Get labels into train_data
        n_row = int(test_data.shape[0] / len(test_files_label))
        test_labels = []
        for i in range(len(test_files_label)):
            test_labels.extend([test_files_label[i] for k in range(n_row)])  # Duplicate label into n_row times
        test_labels = np.array(test_labels)

        if test_data.shape[0] != len(test_labels):
            raise("Test data and labels do not have the same size")

        dataset = DCASE_Dataset(train_data=train_data, train_labels=train_labels,
                                test_data=test_data, test_labels=test_labels,
                                normal_class=0)

        # Initialize DeepSVDD model and set neural network \phi
        deep_SVDD = DeepSVDD(cfg.settings['objective'], cfg.settings['nu'])
        deep_SVDD.set_network(net_name)

        # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
        if load_model:
            deep_SVDD.load_model(model_path=load_model, load_ae=True)
            logger.info('Loading model from %s.' % load_model)

        logger.info('Pretraining: %s' % pretrain)
        if pretrain:
            # Log pretraining details
            logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
            logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
            logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
            logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
            logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
            logger.info('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

            # Pretrain model on dataset (via autoencoder)
            deep_SVDD.pretrain(dataset,
                               optimizer_name=cfg.settings['ae_optimizer_name'],
                               lr=cfg.settings['ae_lr'],
                               n_epochs=cfg.settings['ae_n_epochs'],
                               lr_milestones=(cfg.settings['ae_lr_milestone']),
                               batch_size=cfg.settings['ae_batch_size'],
                               weight_decay=cfg.settings['ae_weight_decay'],
                               device=device,
                               n_jobs_dataloader=n_jobs_dataloader)

        # Log training details
        logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
        logger.info('Training learning rate: %g' % cfg.settings['lr'])
        logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
        logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
        logger.info('Training batch size: %d' % cfg.settings['batch_size'])
        logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

        # Train model on dataset
        deep_SVDD.train(dataset,
                        optimizer_name=cfg.settings['optimizer_name'],
                        lr=cfg.settings['lr'],
                        n_epochs=cfg.settings['n_epochs'],
                        lr_milestones=(cfg.settings['lr_milestone']),
                        batch_size=cfg.settings['batch_size'],
                        weight_decay=cfg.settings['weight_decay'],
                        device=device,
                        n_jobs_dataloader=n_jobs_dataloader)

        # Test model
        deep_SVDD.test(dataset, device=device, n_jobs_dataloader=n_jobs_dataloader)

        # Plot most anomalous and most normal (within-class) test samples
        labels, scores = zip(*deep_SVDD.results['test_scores'])
        labels, scores = np.array(labels), np.array(scores)

        # Reshape score for errors to match each file
        scores_file = scores.reshape(len(test_files_label), n_row)
        # Calculate score of each file
        errors = np.mean(scores_file, axis=1)
        auc = metrics.roc_auc_score(test_files_label,errors)
        print("AUC is: {}".format(auc))
        p_auc = metrics.roc_auc_score(test_files_label, errors, max_fpr=param["max_fpr"])
        print("P_AUC is: {}".format(p_auc))

        anomaly_score_list = []
        for file_idx, file_path in tqdm(enumerate(test_files), total=len(test_files)):
            anomaly_score_list.append([os.path.basename(file_path), errors[file_idx]])

        os.makedirs(param["result_directory_SVDD"], exist_ok=True)
        anomaly_score_csv = "{result}/anomaly_score_{machine_type}.csv".format(result=param["result_directory_SVDD"],
                                                                               machine_type=machine_type)

        com.save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
        com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv))

        #idx_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # sorted from lowest to highest anomaly score

        # Save results, model, and configuration
        deep_SVDD.save_results(export_json=xp_path + '/results.json')
        deep_SVDD.save_model(export_model=xp_path + '/model.tar')
        #cfg.save_config(export_json=xp_path + '/config.json')
