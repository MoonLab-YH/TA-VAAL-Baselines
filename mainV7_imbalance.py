
# Python
import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as T
import torchvision.models as models
import argparse
# Custom
import models.resnet as resnet
from models.resnet import vgg11
from models.query_models import LossNet
from train_test_V2 import train2, test2
from load_dataset import load_dataset
from selection_methods_V2 import *
from config import *


parser = argparse.ArgumentParser()
parser.add_argument("-l","--lambda_loss",type=float, default=1.2,
                    help="Adjustment graph loss parameter between the labeled and unlabeled")
parser.add_argument("-s","--s_margin", type=float, default=0.1,
                    help="Confidence margin of graph")
parser.add_argument("-n","--hidden_units", type=int, default=128,
                    help="Number of hidden units of the graph")
parser.add_argument("-r","--dropout_rate", type=float, default=0.3,
                    help="Dropout rate of the graph neural network")
parser.add_argument("-d","--dataset", type=str, default="cifar10im",
                    help="")
parser.add_argument("-e","--no_of_epochs", type=int, default=200,
                    help="Number of epochs for the active learner")
parser.add_argument("-m","--method_type", type=str, default="TA-VAAL",
                    help="")
parser.add_argument("-c","--cycles", type=int, default=10,
                    help="Number of active learning cycles")
parser.add_argument("-t","--total", type=bool, default=False,
                    help="Training on the entire dataset")
parser.add_argument("-g","--gpu-id", type=int, default=0)

args = parser.parse_args()

##
# Main
if __name__ == '__main__':

    random_seed = 30
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)
    torch.set_num_threads(2)

    method = args.method_type
    methods = ['Random', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss','VAAL','TA-VAAL']
    datasets = ['cifar10','cifar10im', 'cifar100', 'fashionmnist','svhn']
    assert method in methods, 'No method %s! Try options %s'%(method, methods)
    assert args.dataset in datasets, 'No dataset %s! Try options %s'%(args.dataset, datasets)
    '''
    method_type: 'Random', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss','VAAL','TA-VAAL'
    '''
    results = open('results_'+str(args.method_type)+"_"+args.dataset +'_main'+str(args.cycles)+str(args.total)+'.txt','w')
    print("Dataset: %s"%args.dataset)
    print("Method type:%s"%method)
    if args.total:
        TRIALS = 1
        CYCLES = 1
    else:
        CYCLES = args.cycles

    for trial in range(TRIALS):

        # Load training and testing dataset
        data_train, data_unlabeled, data_test, adden, NO_CLASSES, no_train = load_dataset(args.dataset)
        print('The entire datasize is {}'.format(len(data_train)))
        ADDENDUM = adden
        NUM_TRAIN = no_train
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)

        if args.total:
            labeled_set= indices
        else:
            labeled_set = indices[:ADDENDUM]
            unlabeled_set = [x for x in indices if x not in labeled_set]

        train_loader = DataLoader(data_train, batch_size=BATCH,
                                    sampler=SubsetRandomSampler(labeled_set),
                                    pin_memory=True, drop_last=True)
        test_loader  = DataLoader(data_test, batch_size=BATCH)
        dataloaders  = {'train': train_loader, 'test': test_loader}

        for cycle in range(CYCLES):

            # Randomly sample 10000 unlabeled data points
            if not args.total:
                random.shuffle(unlabeled_set)
                subset = unlabeled_set[:SUBSET]
            # Model - create new instance for every cycle so that it resets
            if args.dataset == "fashionmnist":
                resnet18    = resnet.ResNet18fm(num_classes=NO_CLASSES).cuda()
            else:
                #resnet18    = vgg11().cuda()
                resnet18    = resnet.ResNet18(num_classes=NO_CLASSES).cuda()
            if method == 'lloss' or method == 'TA-VAAL':
                #loss_module = LossNet(feature_sizes=[16,8,4,2], num_channels=[128,128,256,512]).cuda()
                loss_module = LossNet().cuda()

            models      = {'backbone': resnet18}
            if method =='lloss' or 'TA-VAAL':
                models = {'backbone': resnet18, 'module': loss_module}
            torch.backends.cudnn.benchmark = True

            # Loss, criterion and scheduler (re)initialization
            criterion      = nn.CrossEntropyLoss(reduction='none')
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)

            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            optimizers = {'backbone': optim_backbone}
            schedulers = {'backbone': sched_backbone}
            if method == 'lloss' or method == 'TA-VAAL':
                optim_module   = optim.SGD(models['module'].parameters(), lr=LR,
                    momentum=MOMENTUM, weight_decay=WDECAY)
                sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)
                optimizers = {'backbone': optim_backbone, 'module': optim_module}
                schedulers = {'backbone': sched_backbone, 'module': sched_module}
            
            # Training and testing
            train2(models, method, criterion, optimizers, schedulers, dataloaders, args.no_of_epochs, EPOCHL)

            acc = test2(models, EPOCH, method, dataloaders, mode='test')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc))
            np.array([method, trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc]).tofile(results, sep=" ")
            results.write("\n")

            # Get the indices of the unlabeled samples to train on next cycle
            arg = query_samples2(models, method, data_unlabeled, subset, labeled_set, cycle, args)
            # save trained models #
            torch.save(models['backbone'].state_dict(), f'Save/TA_VAAL_Backbone_{cycle}Cycle.pth')
            print(f'model saved at Save/TA_VAAL_Backbone_{cycle}Cycle')

            if cycle == (CYCLES - 1):
                # Reached final training cycle
                print("Finished.")
                break

            # Update the labeled dataset and the unlabeled dataset, respectively
            new_list = list(torch.tensor(subset)[arg][:ADDENDUM].numpy())
            # print(len(new_list), min(new_list), max(new_list))
            labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
            listd = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) 
            unlabeled_set = listd + unlabeled_set[SUBSET:]
            print(len(labeled_set), min(labeled_set), max(labeled_set))
            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(data_train, batch_size=BATCH,
                                            sampler=SubsetRandomSampler(labeled_set),
                                            pin_memory=True)

    results.close()