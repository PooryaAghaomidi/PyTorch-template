# -*- coding: utf-8 -*-
""" main.py """

import os
import warnings
from model.model import Model
from train.train import TrainModel
from test.test_model import test_model
from configs.config import CFG
from utils.set_seed import set_seed
from utils.set_device import set_gpu
from utils.per_epoch import batches_per_epoch
from utils.callbacks import callback
from dataloader.dataloader import DataGenerator
from dataset.preprocessing import norm_data

warnings.filterwarnings('ignore')


def run(train_par, data_par, test_par, mode):
    device = set_gpu()
    generator = set_seed(device, seed_value=42)

    datasets = os.listdir('dataset/')
    if ('train.npy' in datasets) and ('test.npy' in datasets) and ('val.npy' in datasets):
        pass
    else:
        norm_data(data_par['train_path'], data_par['test_path'], data_par['validation_path'])

    model = Model(train_par['cls_num']).to(device)
    train_per_epoch = batches_per_epoch('dataset/train.npy', train_par['batch_size'])
    test_per_epoch = batches_per_epoch('dataset/test.npy', train_par['batch_size'])
    val_per_epoch = batches_per_epoch('dataset/val.npy', train_par['batch_size'])

    if train_par['loss'] == 'categorical_crossentropy':
        from loss.categorical_crossentropy import cross_entropy
        my_loss = cross_entropy(label_smoothing=0.0)
    else:
        warnings.warn("The loss is invalid")

    if train_par['optimizer'] == 'adam':
        from optimizer.adam import adam_opt
        my_opt = adam_opt(model, train_par['learning_rate'], weight_decay=0)
    else:
        warnings.warn("The optimizer is invalid")

    callbacks, model_name = callback()
    train_gen = DataGenerator('dataset/train.npy', data_par['shape'], train_par['batch_size'], train_par['cls_num'],
                              device)
    test_gen = DataGenerator('dataset/test.npy', data_par['shape'], train_par['batch_size'], train_par['cls_num'],
                             device)
    val_gen = DataGenerator('dataset/val.npy', data_par['shape'], train_par['batch_size'], train_par['cls_num'],
                            device)

    train_class = TrainModel(model, train_gen, val_gen, my_opt, my_loss, train_par['num_epochs'], train_par['batch_size'],
                             train_per_epoch, val_per_epoch, callbacks, model_name, train_per_epoch // train_par['info_interval'])

    if mode == 'train':
        train_class.training()
        test_model(Model(train_par['cls_num']).to(device), model_name, test_gen, test_per_epoch)

    elif mode == 'test':
        test_model(Model(train_par['cls_num']).to(device), test_par['model_path'], test_gen, test_per_epoch)

    else:
        warnings.warn("The mode is invalid")


if __name__ == '__main__':
    mode = 'train'
    train_par = CFG['train']
    data_par = CFG['data']
    test_par = CFG['test']

    run(train_par, data_par, test_par, mode)
