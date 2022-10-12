#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun, 8 Nov 2020
@author: zxh
"""
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
import numpy as np
import datetime
import os
import sys
from tqdm import tqdm
sys.path.insert(0, '..')
sys.path.append("..")
import copy
import argparse
import logging
import random
import torch.nn as nn
try:
    import cPickle as pickle
except:
    import pickle
import utils_pytorch
from resnet_cifar import resnet32
from train import train
from models import Resnet32
import modified_linear
######### Modifiable Settings ##########
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=4, help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
parser.add_argument('--dataset', default='cifar100', type=str)
parser.add_argument('--method', default='fedavg', type=str)
parser.add_argument('--num_classes', default=100, type=int)
parser.add_argument('--users', default=10, type=int)
parser.add_argument('--nb_cl_fg', default=50, type=int, help='the number of classes in first group')
parser.add_argument('--nb_cl', default=10, type=int, help='Classes per group')
parser.add_argument('--nb_runs', default=1, type=int, help='Number of runs (random ordering of classes at each run)')
parser.add_argument('--rounds', default=5, type=int, help='Number of Rounds')
parser.add_argument('--train_bs', default=128, type=int, help='Batch size for train')
parser.add_argument('--eval_bs', default=128, type=int, help='Batch size for eval')
parser.add_argument('--test_bs', default=100, type=int, help='Batch size for test')
parser.add_argument('--lr', default=0.1, type=float, help='Initial learning rate')
parser.add_argument('--local_lr', default=0.005, type=float, help='Initial learning rate')
parser.add_argument('--lr_strat', default=[80, 120], help='Epochs where learning rate gets decreased')
parser.add_argument('--lr_factor', default=0.1, type=float, help='Learning rate decrease factor')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight Decay')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
parser.add_argument('--ckp_prefix', default=os.path.basename(sys.argv[0])[:-3], type=str, help='Checkpoint prefix')
parser.add_argument('--epochs', default=160, type=int, help='Epochs')
parser.add_argument('--resume', action='store_true',help='resume from checkpoint')
parser.add_argument('--log_dir', default='./log', type=str, help='log dir')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--mu', type=float, default=0.02, help='fedprox mu')
parser.add_argument('--random_seed', type=int, default=1993, help='random seed')
args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
np.random.seed(seed)
random.seed(seed)


# logger
########################################
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

timestr = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

log_save_dir = os.path.join(args.log_dir, args.ckp_prefix + timestr + '.txt')
fh = logging.FileHandler(log_save_dir)
fh.setLevel(logging.INFO)
logger.addHandler(fh)

h1 = logging.StreamHandler(sys.stdout)
logger.addHandler(h1)
logger.info(timestr)
logger.info("Using GPU {} ".format(args.gpu))
logger.info(args)
########################################
if args.gpu:
    torch.cuda.set_device(int(args.gpu))
device = 'cuda' if args.gpu else 'cpu'
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071,  0.4866,  0.4409), (0.2009,  0.1984,  0.2023)),
])
trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform_test)
evalset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=False, transform=transform_test)

def user_split(dataset, users):
    num_items = int(len(dataset)/users)
    num = len(dataset)
    dict_users, all_idxs = {}, list(range(num))
    for i in range(users):
        dict_users[i] = list(set(np.random.choice(all_idxs, num_items, replace=False)))
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
    return dict_users
def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

weights = []
# Initialization
X_train_total = np.array(trainset.data)
Y_train_total = np.array(trainset.targets)
X_valid_total = np.array(testset.data)
Y_valid_total = np.array(testset.targets)

# Launch the different runs
for iteration_total in range(args.nb_runs):
    # Select the order for the class learning
    order_name = "./checkpoint/seed_{}_{}_order_run_{}.pkl".format(args.random_seed, args.dataset, iteration_total)
    logger.info("Order name:{}".format(order_name))
    if os.path.exists(order_name):
        logger.info("Loading orders")
        order = utils_pytorch.unpickle(order_name)
    else:
        logger.info("Generating orders")
        order = np.arange(args.num_classes)
        np.random.shuffle(order)
        utils_pytorch.savepickle(order, order_name)
    order_list = list(order)
    logger.info(order_list)

    # Initialization of the variables for this run
    X_valid_cumuls = []
    X_protoset_cumuls = []
    X_train_cumuls = []
    Y_valid_cumuls = []
    Y_protoset_cumuls = []
    Y_train_cumuls = []
    start_iter = int(args.nb_cl_fg/args.nb_cl)
    train_acc_dict, test_acc_dict = dict(), dict()
    train_acc_dict[0] = list()
    # Prepare the training data for the current batch of classes
    actual_cl = order[range(0, args.nb_cl_fg)]
    indices_train_10 = np.array([i in actual_cl for i in Y_train_total])
    indices_test_10 = np.array([i in actual_cl for i in Y_valid_total])
    X_train = X_train_total[indices_train_10]
    X_valid = X_valid_total[indices_test_10]
    X_valid_cumuls.append(X_valid)
    X_train_cumuls.append(X_train)
    X_valid_cumul = np.concatenate(X_valid_cumuls)
    X_train_cumul = np.concatenate(X_train_cumuls)
    Y_train = Y_train_total[indices_train_10]
    Y_valid = Y_valid_total[indices_test_10]
    Y_valid_cumuls.append(Y_valid)
    Y_train_cumuls.append(Y_train)
    Y_valid_cumul = np.concatenate(Y_valid_cumuls)
    Y_train_cumul = np.concatenate(Y_train_cumuls)

    lr_strat = [80, 120]
    args.epochs = 160
    inc = 0
    eq_start = True
    tg_model = resnet32(num_classes=args.nb_cl_fg)
    in_features = tg_model.fc.in_features
    out_features = tg_model.fc.out_features
    logger.info("in_features:{} out_features:{}".format(in_features, out_features))
    ref_model = None
    logger.info('Batch of classes number {0} arrives ...'.format(args.nb_cl_fg))
    map_Y_train = np.array([order_list.index(i) for i in Y_train])
    map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
    trainset.data = X_train.astype('uint8')
    trainset.targets = map_Y_train
    # trainset.data = X_train_cumul.astype('uint8')
    # map_Y_train_cumul = np.array([order_list.index(i) for i in Y_train])
    # trainset.targets = map_Y_train_cumul
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bs, shuffle=True, num_workers=2)
    testset.data = X_valid_cumul.astype('uint8')
    testset.targets = map_Y_valid_cumul
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_bs, shuffle=False, num_workers=2)
    logger.info('Max and Min of train labels: {}, {}'.format(min(map_Y_train), max(map_Y_train)))
    logger.info('Max and Min of valid labels: {}, {}'.format(min(map_Y_valid_cumul), max(map_Y_valid_cumul)))

    ckp_dir = './checkpoint/{}/'.format(args.ckp_prefix)
    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir)
    ckp_name = './checkpoint/{}/run_{}_iteration_{}_model.pth'.format(args.ckp_prefix, iteration_total,args.nb_cl_fg)

    if args.resume and os.path.exists(ckp_name):
        logger.info("###############################")
        logger.info("Loading models from checkpoint")
        tg_model = torch.load(ckp_name)
        # tg_model.avgpool = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
        logger.info("###############################")
    else:
        tg_params = tg_model.parameters()
        tg_model = tg_model.to(device)
        tg_optimizer = optim.SGD(tg_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat, gamma=args.lr_factor)
        round = 0
        tg_model, train_acc_dict[0], test_acc_dict[0] = train(args.method, args.epochs, round, tg_model, ref_model,tg_optimizer, tg_lr_scheduler, trainloader, testloader,
                                                                              args.mu, logger=logger, ckp_name=ckp_name, device='cuda' if args.gpu else 'cpu')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(tg_model, ckp_name)

    for iteration in range(start_iter, int(args.num_classes / args.nb_cl)):
        last_iter = iteration
        args.epochs = 10
        lr_strat = [1000000]

        in_features = tg_model.fc.in_features
        out_features = tg_model.fc.out_features
        logger.info("in_features:{} out_features:{}".format(in_features, out_features))
        new_fc = nn.Linear(in_features, out_features+args.nb_cl)
        new_fc.weight.data[:out_features] = tg_model.fc.weight.data
        new_fc.bias.data[:out_features] = tg_model.fc.bias.data
        tg_model.fc = new_fc
        new_in_features = tg_model.fc.in_features
        new_out_features = tg_model.fc.out_features
        logger.info("new_in_features:{} new_out_features:{}".format(new_in_features, new_out_features))

        # Prepare the training data for the current batch of classes
        actual_cl        = order[range(last_iter*args.nb_cl,(iteration+1)*args.nb_cl)]
        indices_train_10 = np.array([i in actual_cl for i in Y_train_total])
        indices_test_10  = np.array([i in actual_cl for i in Y_valid_total])
        X_train = X_train_total[indices_train_10]
        X_valid = X_valid_total[indices_test_10]
        X_valid_cumuls.append(X_valid)
        # X_train_cumuls.append(X_train)
        X_valid_cumul = np.concatenate(X_valid_cumuls)
        # X_train_cumul = np.concatenate(X_train_cumuls)
        Y_train = Y_train_total[indices_train_10]
        Y_valid = Y_valid_total[indices_test_10]
        Y_valid_cumuls.append(Y_valid)
        # Y_train_cumuls.append(Y_train)
        Y_valid_cumul = np.concatenate(Y_valid_cumuls)
        # Y_train_cumul = np.concatenate(Y_train_cumuls)
        map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
        testset.data = X_valid_cumul.astype('uint8')
        testset.targets = map_Y_valid_cumul
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_bs, shuffle=False, num_workers=2)

        user_groups = user_split(X_train, args.users)
        user_weights = []
        tg_params = tg_model.parameters()
        tg_model = tg_model.to(device)

        tg_optimizer = optim.SGD(tg_params, lr=args.local_lr, momentum=args.momentum, weight_decay=args.weight_decay)
        tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat, gamma=args.lr_factor)

        if iteration == start_iter:
            X_train_cumuls_id, Y_train_cumuls_id = {}, {}
            for id in range(args.users):
                X_train_cumuls_id[id] = X_train_cumul
                Y_train_cumuls_id[id] = Y_train_cumul



        for round in tqdm(range(args.rounds)):
            logger.info('~~~~~~~~~Round %d'%(round))
            for id in range(args.users):
                user_X_train = X_train[user_groups[id]]
                user_Y_train = Y_train[user_groups[id]]
                # user_map_Y_train = np.array([order_list.index(i) for i in user_Y_train])
                # trainset.data = user_X_train.astype('uint8')
                # trainset.targets = user_map_Y_train

                if round == 0:
                    X_train_cumuls_id[id] = np.concatenate([X_train_cumuls_id[id], user_X_train])
                    Y_train_cumuls_id[id] = np.concatenate([Y_train_cumuls_id[id], user_Y_train])
                    user_map_Y_train_cumul = np.array([order_list.index(i) for i in Y_train_cumuls_id[id]])
                trainset.data = X_train_cumuls_id[id].astype('uint8')
                trainset.targets = user_map_Y_train_cumul

                logger.info('Batch of classes number {0}...'.format(iteration + 1))
                logger.info('user id: {0} | data_size: '.format({len(trainset.data)}))
                # logger.info('Max and Min of train labels: {}, {}'.format(min(user_map_Y_train), max(user_map_Y_train)))
                logger.info('Max and Min of train labels: {}, {}'.format(min(user_map_Y_train_cumul), max(user_map_Y_train_cumul)))
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bs, shuffle=True,num_workers=2)
                user_model, _, _  = train(args.method, args.epochs, round, tg_model,
                                                                      ref_model, tg_optimizer, tg_lr_scheduler,
                                                                      trainloader, testloader,
                                                                      args.mu, logger=logger, ckp_name=ckp_name,
                                                                      device='cuda' if args.gpu else 'cpu')
                user_weights.append(copy.deepcopy(user_model.state_dict()))

            global_weights = average_weights(user_weights)
            tg_model.load_state_dict(global_weights)
            ref_model = copy.deepcopy(tg_model)
            ref_model = ref_model.to(device)

            # eval
            tg_model.eval()
            test_loss = 0
            correct = 0
            total = 0
            criteria = nn.CrossEntropyLoss()
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = tg_model(inputs)
                    loss = criteria(outputs, targets).to(device)
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                logger.info('########################################Global iteration {}!! Test set: {} Test Loss: {:.4f} Acc: {:.4f}'.format( iteration,
                    len(testloader), test_loss / (batch_idx + 1), 100. * correct / total))
            test_acc_dict[iteration] = 100. * correct / total


    np.save('cum_test_acc_dict_fed.npy', test_acc_dict)

