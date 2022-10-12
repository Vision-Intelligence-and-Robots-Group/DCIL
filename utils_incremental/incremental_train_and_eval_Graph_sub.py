#!/usr/bin/env python
# coding=utf-8
import sys
sys.path.extend(['./','../'])
from torch.nn import functional as F
from utils_pytorch import *
from utils_incremental.compute_features import compute_features
from utils_cifar.utils import gen_graph, save_graph, similarity_graph, k_nearest_neighbor
import copy
from utils_incremental.pearson_loss import pearson_loss, pearson_loss_withmask, ehg_pearson_loss, RKdAngle

cur_features = []
ref_features = []
old_scores = []
new_scores = []
def get_ref_features(self, inputs, outputs):
    global ref_features
    ref_features = inputs[0]

def get_cur_features(self, inputs, outputs):
    global cur_features
    cur_features = inputs[0]

def get_old_scores_before_scale(self, inputs, outputs):
    global old_scores
    old_scores = outputs

def get_new_scores_before_scale(self, inputs, outputs):
    global new_scores
    new_scores = outputs

def incremental_train_and_eval_Graph_sub(args, epochs, tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
            trainloader, testloader, \
            iteration, start_iteration, \
            lamda, \
            dist, K, lw_mr, \
            fix_bn=False, weight_per_class=None, device=None, logger=None, ckp_name=None,
            anchorset=None, ckp_prefix='default',
            num_features=64, ehg_bmu=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if iteration > start_iteration:
        ref_model.eval()
        num_old_classes = ref_model.fc.out_features
        handle_ref_features = ref_model.fc.register_forward_hook(get_ref_features)
        handle_cur_features = tg_model.fc.register_forward_hook(get_cur_features)
        handle_old_scores_bs = tg_model.fc.fc1.register_forward_hook(get_old_scores_before_scale)
        handle_new_scores_bs = tg_model.fc.fc2.register_forward_hook(get_new_scores_before_scale)
        anchorloader = torch.utils.data.DataLoader(anchorset, batch_size=100, shuffle=True, num_workers=2)
        anchoriter = iter(anchorloader)
    
    best_acc = 0
    for epoch in range(epochs):
        #train
        tg_model.train()
        # if fix_bn:
        #     for m in tg_model.modules():
        #         if isinstance(m, nn.BatchNorm2d):
        #             m.track_running_stats = False

        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        correct = 0
        total = 0
        tg_lr_scheduler.step()
        logger.info('\nEpoch: %d, LR: %s' % (epoch, str(tg_lr_scheduler.get_lr())))

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            tg_optimizer.zero_grad()
            if iteration == start_iteration:
                outputs = tg_model(inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)
            else:
                try:
                    inputsANC, targetsANC = anchoriter.next()
                except:
                    anchoriter = iter(anchorloader)
                    inputsANC, targetsANC = anchoriter.next()

                anchor_inputs = inputsANC.to(device)
                anchor_targets = targetsANC.to(device)
                #################################################
                outputs = tg_model(anchor_inputs)
                ref_outputs = ref_model(anchor_inputs)
                graph_lambda = args.graph_lambda
                loss4 = pearson_loss(ref_features, cur_features) * graph_lambda
                # loss4 = RKdAngle(cur_features, ref_features) * graph_lambda
                outputs = tg_model(inputs)
                ref_outputs = ref_model(inputs)
                loss2 = nn.CrossEntropyLoss(weight_per_class)(outputs, targets) * args.cls_weight
                loss = loss2 + loss4

            loss.backward()
            tg_optimizer.step()

            train_loss += loss.item()
            if iteration > start_iteration:
                train_loss2 += loss2.item() / args.cls_weight
                train_loss4 += loss4.item() / graph_lambda
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        if iteration == start_iteration:
            logger.info('Train set: {}, Train Loss: {:.4f} Acc: {:.4f}'.format(\
                len(trainloader), train_loss/(batch_idx+1), 100.*correct/total))
        else:
            logger.info('Train set: {}, Train CrossEntropyLoss: {:.4f}, Train Pearson_Loss: {:.4f}'
                        '\n Train Loss: {:.4f} Acc: {:.4f}'.format(len(trainloader),
                train_loss2/(batch_idx+1), train_loss4/(batch_idx+1),
                train_loss/(batch_idx+1), 100.*correct/total))

        #eval
        tg_model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = tg_model(inputs)
                loss = nn.CrossEntropyLoss(weight_per_class)(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        logger.info('Test set: {} Test Loss: {:.4f} Acc: {:.4f}'.format(\
            len(testloader), test_loss/(batch_idx+1), 100.*correct/total))

    if iteration > start_iteration:
        logger.info("Removing register_forward_hook")
        handle_ref_features.remove()
        handle_cur_features.remove()
        handle_old_scores_bs.remove()
        handle_new_scores_bs.remove()
    # user_model = copy.deepcopy(tg_model)
    # return user_model