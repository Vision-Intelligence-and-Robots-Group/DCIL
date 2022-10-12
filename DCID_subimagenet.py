#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun, 15 Dec 2020
@author: zxh
"""
import pdb
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torch.nn import functional as F
import numpy as np
import datetime
import os
import sys
from tqdm import tqdm, trange

sys.path.insert(0, '..')
sys.path.append("..")
import copy
import argparse
import logging
import random
import torch.nn as nn
import modified_linear
import utils_pytorch
import modified_resnet_subimagenet
from utils_incremental.compute_features import compute_features
from utils_incremental.compute_accuracy import compute_accuracy
from utils_incremental.incremental_train_and_eval_Graph import incremental_train_and_eval_Graph
from utils_incremental.incremental_train_and_eval_Graph_sub import incremental_train_and_eval_Graph_sub
from utils_incremental.incremental_train_and_eval_lucir import incremental_train_and_eval_lucir
from utils_incremental.incremental_train_and_eval_icarl import incremental_train_and_eval_icarl
from utils_incremental.incremental_train_and_eval import incremental_train_and_eval_prox
import copy
import datetime
import test

from dataload import large_dataset as subimagenet

try:
    import cPickle as pickle
except:
    import pickle
import utils_pytorch

######### Modifiable Settings ##########
parser = argparse.ArgumentParser()
# parser.add_argument('--gpu', default='1', help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
parser.add_argument('--dataset', default='subimagenet', type=str)
parser.add_argument('--method', default='fedavg', type=str)
parser.add_argument('--iid', default='iid', type=str)
parser.add_argument('--num_classes', default=100, type=int)
parser.add_argument('--users', default=10, type=int)
parser.add_argument('--nb_protos', default=4, type=int, help='Number of prototypes per class at the end')
parser.add_argument('--nb_cl_fg', default=50, type=int, help='the number of classes in first group')
parser.add_argument('--nb_cl', default=10, type=int, help='Classes per group')
parser.add_argument('--nb_runs', default=1, type=int, help='Number of runs (random ordering of classes at each run)')
parser.add_argument('--rounds', default=2, type=int, help='Number of Rounds')
parser.add_argument('--train_bs', default=128, type=int, help='Batch size for train')
parser.add_argument('--eval_bs', default=128, type=int, help='Batch size for eval')
parser.add_argument('--test_bs', default=50, type=int, help='Batch size for test')
parser.add_argument('--lr', default=5e-4, type=float, help='Initial learning rate')
parser.add_argument('--local_lr', default=5e-4, type=float, help='Initial learning rate')
parser.add_argument('--lr_strat', default=[80, 120], help='Epochs where learning rate gets decreased')
parser.add_argument('--lr_factor', default=0.1, type=float, help='Learning rate decrease factor')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight Decay')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
parser.add_argument('--ckp_prefix', type=str, help='Checkpoint prefix')
parser.add_argument('--base_epochs', default=160, type=int, help='Epochs')
parser.add_argument('--epochs', default=5, type=int, help='Epochs')
parser.add_argument('--kd_epochs', default=5, type=int, help='KD_Epochs')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.add_argument('--log_dir', default=os.path.join('./log', 'subimagenet_TPCIL'), type=str, help='log dir')
# parser.add_argument('--seed', type=int, default=4, help='random seed')
parser.add_argument('--prox_mu', type=float, default=0.02, help='fedprox mu')
parser.add_argument('--random_seed', type=int, default=1994, help='random seed')
parser.add_argument('--fix_budget', action='store_true', help='fix budget')
parser.add_argument('--imprint_weights', action='store_true', help='Imprint the weights for novel classes')
parser.add_argument('--lamda', default=5, type=float, help='Lamda for LF')
parser.add_argument('--adapt_lamda', action='store_true', help='Adaptively change lamda')
parser.add_argument('--dist', default=0.5, type=float, help='Dist for MarginRankingLoss')
parser.add_argument('--K', default=2, type=int, help='K for MarginRankingLoss')
parser.add_argument('--lw_mr', default=1, type=float, help='loss weight for margin ranking loss')
########################################
parser.add_argument('--graph_lambda', default=10, type=float)  # pearson=10
parser.add_argument('--ref_nn', default=1, type=int)
parser.add_argument('--cls_weight', default=1, type=float)
parser.add_argument('--herding', default=1)
parser.add_argument('--cil_method', default='tpcil')
parser.add_argument('--pubdataset', default=20, type=int)
args = parser.parse_args()

seed = args.random_seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

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
logger.info(args)
########################################
train_batch_size       = 256            # Batch size for train
test_batch_size        = 50             # Batch size for test
eval_batch_size        = 128            # Batch size for eval
base_lr                = 0.1            # Initial learning rate
lr_strat               = [30, 60]       # Epochs where learning rate gets decreased
lr_factor              = 0.1            # Learning rate decrease factor
custom_weight_decay    = 1e-4           # Weight Decay
custom_momentum        = 0.9            # Momentum
fc_lr = 0.0  # FC layer learning rate
cur_lamda = args.lamda  # 5.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform_anchor = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

print('Load data...')
trainset = subimagenet(root='/data1/zxh/subImageNet', train=True, transform=transform_train)  # train=True
print('done')
testset = subimagenet(root='/data1/zxh/subImageNet', train=False, transform=transform_test)
evalset = subimagenet(root='/data1/zxh/subImageNet', train=False, transform=transform_test)
eval1set = subimagenet(root='/data1/zxh/subImageNet', train=False, transform=transform_test)
kdset = subimagenet(root='/data1/zxh/subImageNet', train=False, transform=transform_anchor)
kdstuset = subimagenet(root='/data1/zxh/subImageNet', train=False, transform=transform_anchor)


def user_split(dataset, users, iteration):
    num_items = int(len(dataset) / users / args.nb_cl)
    # num_items = int(len(dataset) / users / 20)
    num = len(dataset)
    dict_users, all_idxs = {}, list(range(num))
    for i in range(users):
        dict_users[i] = []
    idx_index = {}
    for i in range(iteration * args.nb_cl, (iteration + 1) * args.nb_cl):
        idx_index[i] = []
        for j in range(num):
            if dataset[j] == i:
                idx_index[i].append(j)
    for cla in range(iteration * args.nb_cl, (iteration + 1) * args.nb_cl):
        for i in range(users):
            temp = list(set(np.random.choice(idx_index[cla], num_items, replace=True)))
            dict_users[i].extend(temp)
            idx_index[cla] = list(set(idx_index[cla]) - set(temp))
    return dict_users


def user_split_niid(dataset, targets, users):
    num_items = int(len(dataset) / users)  # 1000
    idx_shard = [i for i in range(users * 2)]  # 50 # 10
    num_imgs = int(num_items / 2)  # 500
    num = len(dataset)
    numidx = np.arange(num)  # [0, ..., 50 00]
    dict_users = {i: [] for i in range(users)}  # {0:[], ..., 5:[]}

    # sort labels
    idxs_labels = np.vstack((numidx, targets))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i].extend(list(idxs[rand * num_imgs:(rand + 1) * num_imgs]))
    return dict_users


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def add_weights(w, beta):
    w_new = copy.deepcopy(w[0])
    for key in w_new.keys():
        for i in range(1, len(w)):
            w_new[key] += w[i][key] * beta
        w_new[key] = torch.div(w_new[key], 1 + beta)
    return w_new


def distillation(student_outputs, targets, teacher_outputs, temp, alpha):
    kl_stu_tea = nn.KLDivLoss()(F.log_softmax(student_outputs / temp, dim=1),
                                F.softmax(teacher_outputs / temp, dim=1)) * temp * temp * 2.0 * alpha
    stu_loss = F.cross_entropy(student_outputs, targets) * (1 - alpha)
    return kl_stu_tea + stu_loss


def train_student_avg_kd(student_model, teacher_outputs, trainloader, optimizer, epochs, device):
    student_model.train()
    # teacher_outputs.to(device)
    for epoch in range(epochs):
        trained_samples = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            student_outputs = student_model(inputs)
            loss = distillation(student_outputs, targets, teacher_outputs[batch_idx], temp=5., alpha=.7)
            loss.backward(retain_graph=True)
            optimizer.step()
            trained_samples += len(inputs)
    return student_model


def train_student_kd(student_model, teacher_model, trainloader, optimizer, epochs, device):
    student_model.train()
    for epoch in range(epochs):
        trained_samples = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            student_outputs = student_model(inputs)
            teacher_outputs = teacher_model(inputs)
            loss = distillation(student_outputs, targets, teacher_outputs, temp=5., alpha=.7)
            loss.backward()
            optimizer.step()
            trained_samples += len(inputs)
    return student_model


def train_student_kd_o2n(student_model, teacher_model, trainloader, optimizer, epochs, device):
    student_model.train()
    for epoch in range(epochs):
        trained_samples = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            student_outputs = student_model(inputs)
            teacher_outputs = teacher_model(inputs)
            loss = distillation(student_outputs[:, :-10], targets, teacher_outputs, temp=5., alpha=.7)
            loss.backward()
            optimizer.step()
            trained_samples += len(inputs)
    return student_model


def train_allstudent_kd(student_model, teacher_outputs, loader_list, optimizer, epochs, device):
    student_model.train()
    teacher_outputs.to(device)
    for epoch in range(epochs):
        trained_samples = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            student_outputs = student_model(inputs)
            loss = distillation(student_outputs, targets, teacher_outputs[batch_idx], temp=5., alpha=1)
            loss.backward(retain_graph=True)
            optimizer.step()
            trained_samples += len(inputs)
    return student_model


weights = []
test_list = []  # every iter server model test
# Initialization
X_train_total = np.array(trainset.data)
Y_train_total = np.array(trainset.targets)
X_valid_total = np.array(testset.data)
Y_valid_total = np.array(testset.targets)
# Initialization
dictionary_size = 1500
top1_acc_list_cumul = np.zeros((int(args.num_classes / args.nb_cl), 3, args.nb_runs))  # (10, 3, 1)
top1_acc_list_ori = np.zeros((int(args.num_classes / args.nb_cl), 3, args.nb_runs))

# Launch the different runs
for iteration_total in range(args.nb_runs):
    # Select the order for the class learning
    iteration = int(args.nb_cl_fg / args.nb_cl) - 1
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

    graph_herding = np.zeros((int(args.num_classes / args.nb_cl), dictionary_size, args.nb_cl),
                             np.float32)  # (10, 500, 10)
    # prototypes = np.zeros((args.num_classes, dictionary_size, X_train_total.shape[1], X_train_total.shape[2],
    #                        X_train_total.shape[3]), dtype=np.float32)  # (100,500,32,32,3)
    prototypes = []
    for orde in range(args.num_classes):
        # prototypes[orde, :, :, :, :] = X_train_total[np.where(Y_train_total == order[orde])]
        # prototypes[orde] = X_train_total[np.where(Y_train_total == order[orde])]
        prototypes.append(X_train_total[np.where(Y_train_total == order[orde])])
    start_iter = int(args.nb_cl_fg / args.nb_cl) - 1  # 4
    last_iter = 0
    inc = 0
    lr_strat = [30, 60]
    ############################################################
    # tg_model = modified_resnet_cifar.resnet32(num_classes=args.nb_cl_fg)
    tg_model = modified_resnet_subimagenet.resnet18(num_classes=args.nb_cl_fg)
    in_features = tg_model.fc.in_features  # 64
    out_features = tg_model.fc.out_features  # 50
    logger.info("in_features: {} out_features: {}".format(in_features, out_features))
    ref_model = None
    actual_cl = order[range(0, args.nb_cl_fg)]
    indices_train_10 = np.array([i in actual_cl for i in Y_train_total])
    indices_test_10 = np.array([i in actual_cl for i in Y_valid_total])
    X_train, Y_train = X_train_total[indices_train_10], Y_train_total[indices_train_10]  # ( , 32, 32, 3), ( ,)
    X_valid, Y_valid = X_valid_total[indices_test_10], Y_valid_total[indices_test_10]  # ( , 32, 32, 3), ( ,)
    X_train_cumuls.append(X_train)
    X_valid_cumuls.append(X_valid)
    X_train_cumul, X_valid_cumul = np.concatenate(X_train_cumuls), np.concatenate(X_valid_cumuls)
    # X_train_cumul, X_valid_cumul = X_train_cumuls, X_valid_cumuls
    Y_valid_cumuls.append(Y_valid)
    Y_train_cumuls.append(Y_train)
    Y_valid_cumul, Y_train_cumul = np.concatenate(Y_valid_cumuls), np.concatenate(Y_train_cumuls)

    X_valid_ori, Y_valid_ori = X_valid, Y_valid

    logger.info('Batch of classes number {0} arrives ...'.format(args.nb_cl_fg))
    map_Y_train = np.array([order_list.index(i) for i in Y_train])
    map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
    trainset.data = X_train
    trainset.targets = map_Y_train
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=8)
    testset.data = X_valid_cumul
    testset.targets = map_Y_valid_cumul
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=8)
    logger.info('Max and Min of train labels: {}, {}'.format(min(map_Y_train), max(map_Y_train)))
    logger.info('Max and Min of valid labels: {}, {}'.format(min(map_Y_valid_cumul), max(map_Y_valid_cumul)))
    ##############################################################
    ckp_dir = './checkpoint/{}/'.format(args.ckp_prefix)
    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir)
    ckp_name = './checkpoint/{}/run_{}_iteration_{}_model.pth'.format(args.ckp_prefix, iteration_total, 4)

    if args.resume and os.path.exists(ckp_name):
        logger.info("############Loading models from checkpoint############")
        tg_model = torch.load(ckp_name, map_location=device)
    else:
        tg_model = tg_model.to(device)
        tg_params = tg_model.parameters()
        tg_optimizer = optim.SGD(tg_params, lr=base_lr, momentum=custom_momentum, weight_decay=custom_weight_decay)
        tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat, gamma=lr_factor)

        logger.info("############incremental_train_and_eval_Graph############")
        tg_model = incremental_train_and_eval_Graph(args, args.base_epochs, tg_model, ref_model, tg_optimizer,
                                                    tg_lr_scheduler, \
                                                    trainloader, testloader, \
                                                    iteration, start_iter, \
                                                    cur_lamda, \
                                                    args.dist, args.K, args.lw_mr, logger=logger,
                                                    ckp_name=ckp_name, device=device)
        torch.save(tg_model, ckp_name)

    ### Exemplars
    nb_protos_cl = args.nb_protos
    tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])  # without fc layer
    num_features = tg_model.fc.in_features  # 64

    # Graphing (graph of each class)
    logger.info('Herding: Updating graph')
    herding_sample = args.herding
    # last_iter start from 0; iteration start from 4.
    if herding_sample:
        for iter_dico in range(last_iter * args.nb_cl, (iteration + 1) * args.nb_cl):  # [0-50)
            print('%d/%d' % (iter_dico + 1, (iteration + 1) * args.nb_cl), end='\r')
            # Possible exemplars in the feature space and projected on the L2 sphere
            evalset.data = prototypes[iter_dico]  # (500, 32, 32, 3)
            evalset.targets = np.zeros(len(evalset.data))  # zero labels
            evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                                                     shuffle=False, num_workers=8)
            num_samples = len(evalset.data)  # 500
            mapped_prototypes = compute_features(tg_feature_model, evalloader, num_samples,
                                                 num_features, device=device)  # (50, 512)
            mapped_prototypes = mapped_prototypes.numpy()
            D = mapped_prototypes.T  # (512, 50)
            D = D / np.linalg.norm(D, axis=0)  # 特征的二范数, (64, 500)
            # Herding procedure : ranking of the potential exemplars
            mu = np.mean(D, axis=1)  # (64, )
            index1 = int(iter_dico / args.nb_cl)
            index2 = iter_dico % args.nb_cl
            graph_herding[index1, :, index2] = graph_herding[index1, :, index2] * 0
            w_t = mu  # (512,)
            iter_herding = 0
            iter_herding_eff = 0
            while not (np.sum(graph_herding[index1, :, index2] != 0) == min(nb_protos_cl,
                                                                            1500)):  # and iter_herding_eff < 1000:
                tmp_t = np.dot(w_t, D)  # (500, )
                ind_max = np.argmax(tmp_t)  # index of max
                iter_herding_eff += 1
                if graph_herding[index1, ind_max, index2] == 0:
                    graph_herding[index1, ind_max, index2] = 1 + iter_herding
                    iter_herding += 1
                w_t = w_t + mu - D[:, ind_max]  # (64, )
            graph_herding = (graph_herding > 0) * (
                    graph_herding < nb_protos_cl + 1) * 1.
            # (10, 500, 10), max=20, min=0, 按顺序选20个标为1-20. 而这一步把所有非零数字都变成1

    # Prepare the protoset
    X_protoset_cumuls = []
    Y_protoset_cumuls = []
    # Class means for iCaRL and NCM + Storing the selected exemplars in the protoset
    logger.info('Computing mean-of_exemplars and theoretical mean...')
    # class_means = np.zeros((2048, 100, 2))
    for iteration2 in range(iteration + 1):
        for iter_dico in range(args.nb_cl):
            # current_cl = order[range(iteration2 * args.nb_cl, (iteration2 + 1) * args.nb_cl)]  # (10, )
            # # Collect data in the feature space for each class
            evalset.data = prototypes[iteration2 * args.nb_cl + iter_dico]  # (500, 32, 32, 3)
            evalset.targets = np.zeros(len(evalset.data))  # zero labels
            evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                                                     shuffle=False, num_workers=8)
            num_samples = len(evalset.data)

            # iCaRL
            alph = graph_herding[iteration2, :, iter_dico]  # (500, )其中20个为1，480个为0, iteration2=十位数， iter_dico=个位数
            assert ((alph[num_samples:] == 0).all())
            alph = alph[:num_samples]
            # X_protoset_cumuls.append(prototypes[iteration2 * args.nb_cl + iter_dico, np.where(alph == 1)[
            #     0]])  # prototypes[classes, samples(20), w(32), h(32), c(3)]
            X_protoset_cumuls.append(prototypes[iteration2 * args.nb_cl + iter_dico][np.where(alph == 1)[0]])
            Y_protoset_cumuls.append(order[iteration2 * args.nb_cl + iter_dico] * np.ones(
                len(np.where(alph == 1)[0])))  # sample(20) label og the class

    ##############################################################
    # Calculate validation error of model on the first nb_cl classes:
    map_Y_valid_ori = np.array([order_list.index(i) for i in Y_valid_ori])
    logger.info('Computing accuracy on the original batch of classes...')
    evalset.data = X_valid_ori
    evalset.targets = map_Y_valid_ori
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                                             shuffle=False, num_workers=8)

    map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
    logger.info('Computing cumulative accuracy...')
    evalset.data = X_valid_cumul
    evalset.targets = map_Y_valid_cumul
    evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                                             shuffle=False, num_workers=8)


    ############################ base classes train over #################################

    ##############################incremental learning###################################
    usertest, testacc_stu, test_newcl_s, test_newcl_u, test_oldcl_s, test_oldcl_u, test_oldcl_old_s = {}, {}, {}, {}, {}, {}, {}
    sec_server_acc = {}
    base_lr = [0.1, 0.01]

    for iteration in range(start_iter + 1, int(args.num_classes / args.nb_cl)):
        usertest[iteration] = []
        testacc_stu[iteration] = []
        test_newcl_s[iteration], test_newcl_u[iteration], test_oldcl_s[iteration], test_oldcl_u[iteration], \
        test_oldcl_old_s[iteration] = [], [], [], [], []

    for iteration in range(start_iter + 1, int(args.num_classes / args.nb_cl)):
        ckp_dir = './checkpoint/{}/'.format(args.ckp_prefix)
        if not os.path.exists(ckp_dir):
            os.makedirs(ckp_dir)
        ckp_name = './checkpoint/{}/run_{}_iteration_{}_model.pth'.format(args.ckp_prefix, iteration_total, iteration)

        if args.resume and os.path.exists(ckp_name):
            logger.info("############Loading models from checkpoint############")
            tg_model = torch.load(ckp_name, map_location=device)
        else:
            last_iter = iteration
            inc = 1
            # lr_strat = [10]
            lr_strat = [10,30,60]

            ref_model = copy.deepcopy(tg_model)
            in_features = tg_model.fc.in_features
            X_protoset_new_id, Y_protoset_new_id = {}, {}

            if iteration == start_iter + 1:
                out_features = tg_model.fc.out_features
                logger.info("in_features: {} out_features: {}".format(in_features, out_features))
                new_fc = modified_linear.SplitCosineLinear(in_features, out_features, args.nb_cl)
                new_fc.fc1.weight.data = tg_model.fc.weight.data
                # new_fc.sigma.data = tg_model.fc.sigma.data
                tg_model.fc = new_fc  # (64, 60)
                lamda_mult = out_features * 1.0 / args.nb_cl

                X_protoset_cumuls_id, Y_protoset_cumuls_id = {}, {}
                user_X_train, user_Y_train, user_map_Y_train = {}, {}, {}
                graph_herding_id = {}
                prototypes_id = {}
                X_pub_id, Y_pub_id = {}, {}

                for idx in range(args.users):
                    X_protoset_cumuls_id[idx] = np.concatenate(X_protoset_cumuls)
                    Y_protoset_cumuls_id[idx] = np.concatenate(Y_protoset_cumuls)
                    X_pub_id[idx] = np.concatenate(X_protoset_cumuls)
                    Y_pub_id[idx] = np.concatenate(Y_protoset_cumuls)
                    graph_herding_id[idx] = graph_herding
                    prototypes_id[idx] = {}

            else:
                out_features1 = tg_model.fc.fc1.out_features
                out_features2 = tg_model.fc.fc2.out_features
                logger.info(
                    "in_features: {} out_features1: {} out_features2: {}".format(in_features, out_features1,
                                                                                 out_features2))
                new_fc = modified_linear.SplitCosineLinear(in_features, out_features1 + out_features2, args.nb_cl)
                new_fc.fc1.weight.data[:out_features1] = tg_model.fc.fc1.weight.data
                new_fc.fc1.weight.data[out_features1:] = tg_model.fc.fc2.weight.data
                # new_fc.sigma.data = tg_model.fc.sigma.data
                tg_model.fc = new_fc
                lamda_mult = (out_features1 + out_features2) * 1.0 / (args.nb_cl)
            new_in_features = tg_model.fc.in_features
            new_out_features = tg_model.fc.out_features
            logger.info("new_in_features:{} new_out_features:{}".format(new_in_features, new_out_features))

            # Prepare the training data for the current batch of classes
            actual_cl = order[range(last_iter * args.nb_cl, (iteration + 1) * args.nb_cl)]
            indices_train_10 = np.array([y in actual_cl for y in Y_train_total])
            indices_test_10 = np.array([i in actual_cl for i in Y_valid_total])

            X_train = X_train_total[indices_train_10]
            X_valid = X_valid_total[indices_test_10]
            X_valid_cumuls.append(X_valid)
            X_valid_cumul = np.concatenate(X_valid_cumuls)
            Y_train = Y_train_total[indices_train_10]
            map_Y_train = np.array([order_list.index(i) for i in Y_train])
            Y_valid = Y_valid_total[indices_test_10]
            Y_valid_cumuls.append(Y_valid)
            Y_valid_cumul = np.concatenate(Y_valid_cumuls)
            map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
            testset.data = X_valid_cumul
            testset.targets = map_Y_valid_cumul

            logger.info(len(map_Y_valid_cumul))

            testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_bs, shuffle=False, num_workers=8)

            for round in tqdm(range(args.rounds)):
                logger.info('~~~~~~~~~Round %d' % (round))

                anchorset = {}
                X_protoset_id, Y_protoset_id = {}, {}
                anchor_X_id, anchor_Y_id, map_anchor_Y_id = {}, {}, {}
                if args.iid == 'iid':
                    user_groups = user_split(map_Y_train, args.users, iteration)
                elif args.iid == 'noniid':
                    user_groups = user_split_niid(X_train, Y_train, args.users)
                else:
                    print("iid or noniid?")
                user_weights = []
                user_model = {}

                devices = {0: 'cuda:0', 1: 'cuda:0', 2: 'cuda:1', 3: 'cuda:1', 4: 'cuda:2'}

                tg_model = tg_model.to(device)
                ref_model = ref_model.to(device)

                for idx in range(args.users):
                    user_model[idx] = copy.deepcopy(tg_model)
                    anchorset[idx] = subimagenet(root='/data1/zxh/subImageNet', train=False, transform=transform_anchor)

                    X_protoset_id[idx] = X_protoset_cumuls_id[idx]  # iter=5: (1000, 32, 32, 3)
                    Y_protoset_id[idx] = Y_protoset_cumuls_id[idx]  # (1000,)
                    user_X_train[idx] = X_train[user_groups[idx]]
                    user_Y_train[idx] = Y_train[user_groups[idx]]

                    for orde in range(last_iter * args.nb_cl, (iteration + 1) * args.nb_cl):  # (50, 60)
                        prototypes_id[idx][orde] = user_X_train[idx][np.where(user_Y_train[idx] == order[orde])]
                    # prototypes_id[idx]:dict{50:[](93,32,32,3), 51:...,59:[]}
                    user_X_train[idx] = np.concatenate(
                        (user_X_train[idx], X_protoset_id[idx]))  # iter=5: (2000, 32, 32, 3)
                    user_Y_train[idx] = np.concatenate((user_Y_train[idx], Y_protoset_id[idx]))  # iter=5: (2000,)
                    user_map_Y_train[idx] = np.array([order_list.index(i) for i in user_Y_train[idx]])
                    trainset.data = user_X_train[idx]
                    trainset.targets = user_map_Y_train[idx]
                    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True,
                                                              num_workers=8)

                    anchor_X_id[idx], anchor_Y_id[idx] = X_protoset_id[idx], Y_protoset_id[idx]  # (1000, 32, 32, 3)
                    map_anchor_Y = np.array([order_list.index(i) for i in anchor_Y_id[idx]])  # (1000, )
                    anchorset[idx].data = anchor_X_id[idx]  # iter=5:(1000, 32, 32, 3)
                    anchorset[idx].targets = map_anchor_Y
                    # Launch the training loop
                    logger.info('Batch of classes number {0} arrives ...'.format(iteration + 1))
                    logger.info(f'user id: {idx + 1} | data_size: {len(trainset.data)}')  # user id:1|data_size:2000

                    # if iteration > start_iter + 1:
                    #     # fix the embedding of old classes
                    #     ignored_params = list(map(id, user_model[idx].fc.fc1.parameters()))
                    #     base_params = filter(lambda p: id(p) not in ignored_params, tg_model.parameters())
                    #     tg_params = [
                    #         {'params': base_params, 'lr': 0.1, 'weight_decay': custom_weight_decay},
                    #         {'params': user_model[idx].fc.fc1.parameters(), 'lr': fc_lr, 'weight_decay': 0}]
                    # else:
                    #     tg_params = user_model[idx].parameters()

                    tg_params = user_model[idx].parameters()
                    if round < 4:
                        tg_optimizer = optim.SGD(tg_params, lr=0.1, momentum=custom_momentum,
                                                 weight_decay=custom_weight_decay)
                        tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat, gamma=lr_factor)
                    else:
                        tg_optimizer = optim.SGD(tg_params, lr=0.01, momentum=custom_momentum,
                                                 weight_decay=custom_weight_decay)
                        tg_lr_scheduler = lr_scheduler.MultiStepLR(tg_optimizer, milestones=lr_strat, gamma=lr_factor)
                    logger.info("############incremental_train_and_eval_Graph############")
                    if args.cil_method == 'icarl':
                        # tmp = test.test(user_model[idx], testloader, logger, device=device)
                        incremental_train_and_eval_icarl(args, args.epochs, user_model[idx], ref_model,
                                                         tg_optimizer, tg_lr_scheduler, \
                                                         trainloader, testloader, \
                                                         iteration, start_iter, \
                                                         cur_lamda, \
                                                         args.dist, args.K, args.lw_mr, logger=logger,
                                                         ckp_name=ckp_name,
                                                         anchorset=anchorset[idx],
                                                         ckp_prefix=args.ckp_prefix,
                                                         num_features=num_features, device=device)
                    elif args.cil_method == 'tpcil':
                        incremental_train_and_eval_Graph_sub(args, args.epochs, user_model[idx], ref_model,
                                                         tg_optimizer, tg_lr_scheduler, \
                                                         trainloader, testloader, \
                                                         iteration, start_iter, \
                                                         cur_lamda, \
                                                         args.dist, args.K, args.lw_mr, logger=logger,
                                                         ckp_name=ckp_name,
                                                         anchorset=anchorset[idx],
                                                         ckp_prefix=args.ckp_prefix,
                                                         num_features=num_features, device=device)
                    elif args.cil_method == 'lucir':
                        incremental_train_and_eval_lucir(args, args.epochs, user_model[idx], ref_model,
                                                         tg_optimizer, tg_lr_scheduler, \
                                                         trainloader, testloader, \
                                                         iteration, start_iter, \
                                                         cur_lamda, \
                                                         args.dist, args.K, args.lw_mr, logger=logger,
                                                         ckp_name=ckp_name,
                                                         anchorset=anchorset[idx],
                                                         ckp_prefix=args.ckp_prefix,
                                                         num_features=num_features, device=device)

                    tmp = test.test(user_model[idx], testloader, logger, device=device)

                    usertest[iteration].append(tmp)

                    user_weights.append(user_model[idx].state_dict())

                global_weights = average_weights(user_weights)
                tg_model.load_state_dict(global_weights)
                # last_model = copy.deepcopy(tg_model)
                # torch.save(tg_model, ckp_name)
                testacc = test.test(tg_model, testloader, logger, device=device)
                test_list.append(testacc)
                logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~iter:{}".format(iteration))
                logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~round:{}".format(round))
                logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~after all users test:{}".format(testacc))

            ###############################exemplars maybe here#################
            for idx in range(args.users):
                nb_protos_cl = args.nb_protos
                user_feature_model = nn.Sequential(*list(user_model[idx].children())[:-1])  # without fc layer
                num_features = user_model[idx].fc.in_features  # 64
                # Graphing (graph of each class)
                logger.info('User {} Herding: Updating graph'.format(idx))

                for iter_dico in range(last_iter * args.nb_cl, (iteration + 1) * args.nb_cl):
                    print('%d/%d' % (iter_dico + 1, (iteration + 1) * args.nb_cl), end='\r')
                    alph_index = []
                    # Possible exemplars in the feature space and projected on the L2 sphere
                    if (len(prototypes_id[idx][iter_dico]) != 0):
                        evalset.data = prototypes_id[idx][iter_dico]  # ex:(93, 32, 32, 3)
                        evalset.targets = np.zeros(len(evalset.data))  # zero labels
                        evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                                                                 shuffle=False, num_workers=8)
                        num_samples = len(evalset.data)  # ex: 93
                        mapped_prototypes = compute_features(user_feature_model, evalloader, num_samples,
                                                             num_features, device=device)  # ex:torch(93, 64)
                        mapped_prototypes = mapped_prototypes.numpy()
                        D = mapped_prototypes.T  # (64, 93)
                        D = D / np.linalg.norm(D, axis=0)  # 特征的二范数, (64, 93)
                        # Herding procedure : ranking of the potential exemplars
                        mu = np.mean(D, axis=1)  # (64, )
                        index1 = int(iter_dico / args.nb_cl)
                        index2 = iter_dico % args.nb_cl
                        graph_herding_id[idx][index1, :, index2] = graph_herding_id[idx][index1, :, index2] * 0
                        w_t = mu  # 64
                        iter_herding = 0
                        iter_herding_eff = 0
                        while not (np.sum(graph_herding_id[idx][index1, :, index2] != 0) == min(nb_protos_cl,
                                                                                                500)) and iter_herding_eff < 1000:
                            tmp_t = np.dot(w_t, D)  # (93, )
                            ind_max = np.argmax(tmp_t)  # index of max
                            iter_herding_eff += 1
                            if graph_herding_id[idx][index1, ind_max, index2] == 0:
                                graph_herding_id[idx][index1, ind_max, index2] = 1 + iter_herding
                                iter_herding += 1
                                alph_index.append(ind_max)
                            w_t = w_t + mu - D[:, ind_max]  # (64, )
                        # graph_herding_id[idx] = (graph_herding_id[idx] > 0) * (graph_herding_id[
                        #                                                            idx] < nb_protos_cl + 1) * 1.  # (10, 500, 10), max=20, min=0, 按顺序选20个标为1-20. 而这一步把所有非零数字都变成1
                        # alph = graph_herding_id[idx][index1, :,
                        #        index2]  # (500, )其中20个为1，480个为0, iteration2=十位数， iter_dico=个位数
                        # alph_index = []  # !=0 index
                    else:
                        alph = np.array([])
                        alph_index = []

                    # for i in range(len(alph)):
                    #     if alph[i] == 1:
                    #         alph_index.append(i)  # [22, 61, 71, 75]

                    for i in range(len(alph_index)):
                        # np.concatenate(
                        Y_protoset_cumuls_id[idx] = Y_protoset_cumuls_id[idx].tolist()
                        X_protoset_cumuls_id[idx] = np.concatenate(
                            (X_protoset_cumuls_id[idx], np.array([prototypes_id[idx][iter_dico][i]])))
                        # X_protoset_cumuls_id[idx]: (1000, ) numpy(tuple); prototypes_id[idx][iter_dico]: (iter_dico=50),dict,[i]numpy_str
                        Y_protoset_cumuls_id[idx].append(order[iter_dico])
                        Y_protoset_cumuls_id[idx] = np.array(Y_protoset_cumuls_id[idx])
                ##############################public dataset########################
                for iter_dico in range(last_iter * args.nb_cl, (iteration + 1) * args.nb_cl):
                    print('%d/%d' % (iter_dico + 1, (iteration + 1) * args.nb_cl), end='\r')
                    alph_index = []
                    # Possible exemplars in the feature space and projected on the L2 sphere
                    if (len(prototypes_id[idx][iter_dico]) != 0):
                        evalset.data = prototypes_id[idx][iter_dico]  # ex:(93, 32, 32, 3)
                        evalset.targets = np.zeros(len(evalset.data))  # zero labels
                        evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                                                                 shuffle=False, num_workers=8)
                        num_samples = len(evalset.data)  # ex: 93
                        mapped_prototypes = compute_features(user_feature_model, evalloader, num_samples,
                                                             num_features, device=device)  # ex:torch(93, 64)
                        mapped_prototypes = mapped_prototypes.numpy()
                        D = mapped_prototypes.T  # (64, 93)
                        D = D / np.linalg.norm(D, axis=0)  # 特征的二范数, (64, 93)
                        # Herding procedure : ranking of the potential exemplars
                        mu = np.mean(D, axis=1)  # (64, )
                        index1 = int(iter_dico / args.nb_cl)
                        index2 = iter_dico % args.nb_cl
                        graph_herding_id[idx][index1, :, index2] = graph_herding_id[idx][index1, :, index2] * 0
                        w_t = mu  # 64
                        iter_herding = 0
                        iter_herding_eff = 0
                        while not (np.sum(graph_herding_id[idx][index1, :, index2] != 0) == min(args.pubdataset,
                                                                                                500)) and iter_herding_eff < 1000:
                            tmp_t = np.dot(w_t, D)  # (93, )
                            ind_max = np.argmax(tmp_t)  # index of max
                            iter_herding_eff += 1
                            if graph_herding_id[idx][index1, ind_max, index2] == 0:
                                graph_herding_id[idx][index1, ind_max, index2] = 1 + iter_herding
                                iter_herding += 1
                                alph_index.append(ind_max)
                            w_t = w_t + mu - D[:, ind_max]  # (64, )
                        # graph_herding_id[idx] = (graph_herding_id[idx] > 0) * (graph_herding_id[
                        #                                                            idx] < nb_protos_cl + 1) * 1.  # (10, 500, 10), max=20, min=0, 按顺序选20个标为1-20. 而这一步把所有非零数字都变成1
                        # alph = graph_herding_id[idx][index1, :,
                        #        index2]  # (500, )其中20个为1，480个为0, iteration2=十位数， iter_dico=个位数
                        # alph_index = []  # !=0 index
                    else:
                        alph = np.array([])
                        alph_index = []

                    # for i in range(len(alph)):
                    #     if alph[i] == 1:
                    #         alph_index.append(i)  # [22, 61, 71, 75]

                    for i in range(len(alph_index)):
                        # np.concatenate(
                        Y_pub_id[idx] = Y_pub_id[idx].tolist()
                        X_pub_id[idx] = np.concatenate(
                            (X_pub_id[idx], np.array([prototypes_id[idx][iter_dico][i]])))
                        # X_protoset_cumuls_id[idx]: (1000, ) numpy(tuple); prototypes_id[idx][iter_dico]: (iter_dico=50),dict,[i]numpy_str
                        Y_pub_id[idx].append(order[iter_dico])
                        Y_pub_id[idx] = np.array(Y_pub_id[idx])
            ################################### knowledge distillation here ######################################
            X_anchor_id, Y_anchor_id, Y_map_anchor_id = {}, {}, {}
            student_model_state_ls = []
            stu_outputs = []
            for idx in range(args.users):
                X_anchor_id[idx] = X_pub_id[idx][-args.nb_cl * args.pubdataset:]  # X anchor of user
                Y_anchor_id[idx] = Y_pub_id[idx][-args.nb_cl * args.pubdataset:]  # Y anchor of user
                Y_map_anchor_id[idx] = np.array([order_list.index(i) for i in Y_anchor_id[idx]])
                if idx == 0:
                    kdstuset.data = X_anchor_id[idx]
                    kdstuset.targets = Y_map_anchor_id[idx]
                else:
                    kdstuset.data = np.concatenate((kdstuset.data, X_anchor_id[idx]))
                    kdstuset.targets = np.concatenate((kdstuset.targets, Y_map_anchor_id[idx]))
            permutation = np.random.permutation(len(kdstuset.data))
            kdstuset.data = kdstuset.data[permutation]
            kdstuset.targets = kdstuset.targets[permutation]
            trainloader = torch.utils.data.DataLoader(kdstuset, batch_size=128, shuffle=False, num_workers=8)

            for idx in range(args.users):
                with torch.no_grad():
                    student_params = user_model[idx].parameters()
                    student_optimizer = optim.SGD(student_params, lr=0.0001, momentum=custom_momentum,
                                                  weight_decay=custom_weight_decay)
                    for batch_idx, (inputs, targets) in enumerate(trainloader):
                        inputs, targets = inputs.to(device), targets.to(device)
                        if idx == 0:
                            stu_outputs.append(user_model[idx](inputs))
                        else:
                            stu_outputs[batch_idx] += user_model[idx](inputs)

            stu_outputs_avg = [i / args.users for i in stu_outputs]  # internal distillation (teacher logits)

            for idx in range(args.users):
                student_params = user_model[idx].parameters()
                student_optimizer = optim.SGD(student_params, lr=0.0001, momentum=custom_momentum,
                                              weight_decay=custom_weight_decay)
                user_model[idx] = train_student_avg_kd(user_model[idx], stu_outputs_avg, trainloader, student_optimizer, \
                                                     args.kd_epochs, device=device)
                test_tmp = test.test(user_model[idx], testloader, logger, device=device)
                testacc_stu[iteration].append(test_tmp)
                student_model_state_ls.append(user_model[idx].state_dict())
            global_weights = average_weights(student_model_state_ls)
            final_model1 = copy.deepcopy(tg_model)
            final_model1.load_state_dict(global_weights)
            test_tmp = test.test(final_model1, testloader, logger, device=device)
            testacc_stu[iteration].append(test_tmp)
            logger.info("Fedavg1 testacc: {}".format(test_tmp))

            student_model = copy.deepcopy(tg_model)
            student_params = student_model.parameters()
            student_optimizer = optim.SGD(student_params, lr=0.0001, momentum=custom_momentum,
                                          weight_decay=custom_weight_decay)
            student_model.train()
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                with torch.no_grad():
                    for idx in range(args.users):
                        inputs, targets = inputs.to(device), targets.to(device)
                        if idx == 0:
                            teacher_outputs = user_model[idx](inputs)
                        else:
                            teacher_outputs += user_model[idx](inputs)
                teacher_outputs /= args.users
                student_optimizer.zero_grad()
                student_outputs = student_model(inputs)
                loss = distillation(student_outputs, targets, teacher_outputs, temp=5., alpha=1)
                loss.backward()
                student_optimizer.step()


            test_tmp = test.test(student_model, testloader, logger, device=device)
            testacc_stu[iteration].append(test_tmp)
            # tg_model = copy.deepcopy(student_model)
            logger.info("Fedavg2 testacc: {}".format(test_tmp))
            #####################inner & outer###############################
            student_model = copy.deepcopy(final_model1)
            # student_model = torch.nn.DataParallel(student_model)
            student_params = student_model.parameters()
            student_optimizer = optim.SGD(student_params, lr=0.0001, momentum=custom_momentum,
                                          weight_decay=custom_weight_decay)
            student_model.train()
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                with torch.no_grad():
                    for idx in range(args.users):
                        user_model[idx].eval()
                        inputs, targets = inputs.to(device), targets.to(device)
                        if idx == 0:
                            teacher_outputs = user_model[idx](inputs)
                        else:
                            teacher_outputs += user_model[idx](inputs)
                teacher_outputs /= args.users
                student_optimizer.zero_grad()
                student_outputs = student_model(inputs)
                loss = distillation(student_outputs, targets, teacher_outputs, temp=5., alpha=1)
                loss.backward()
                student_optimizer.step()

            test_tmp = test.test(student_model, testloader, logger, device=device)
            testacc_stu[iteration].append(test_tmp)
            # tg_model = copy.deepcopy(student_model)
            logger.info("Fedavg3 testacc: {}".format(test_tmp))

            ############################## after inter & outer kd test acc ###########################
            # eval1set.data = X_valid_cumul[-500:]  # test new
            # eval1set.targets = map_Y_valid_cumul[-500:]
            # eval1loader = torch.utils.data.DataLoader(eval1set, batch_size=eval_batch_size,
            #                                           shuffle=False, num_workers=2)
            # test_newcl_s[iteration].append(
            #     test.test(student_model, eval1loader, logger, device=device))
            #
            # evalset.data = X_valid_cumul[: -500]  # test old
            # evalset.targets = map_Y_valid_cumul[: -500]
            # evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
            #                                          shuffle=False, num_workers=2)
            # test_oldcl_s[iteration].append(
            #     test.test(student_model, evalloader, logger, device=device))
            ###################################################################################

        logger.info("fedavg_test:{}".format(test_list))
        logger.info("user_test:{}".format(usertest))
        logger.info("server_test(after teacher):{}".format(testacc_stu))
    logger.info("######################Final########################")
    logger.info("fedavg_test:{}".format(test_list))
    logger.info("user_test:{}".format(usertest))
    # logger.info("test_newcl_s:{}".format(test_newcl_s))
    # logger.info("test_newcl_u:{}".format(test_newcl_u))
    # logger.info("test_oldcl_s:{}".format(test_oldcl_s))
    # logger.info("test_oldcl_u:{}".format(test_oldcl_u))
    # logger.info("test_oldcl_old_s:{}".format(test_oldcl_old_s))
    logger.info("server_test(after teacher):{}".format(testacc_stu))
