
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
import os
# Custom
import models.resnet as resnet
from models.resnet import vgg11
from models.query_models import LossNet
from train_test_V2 import train2, test2, train_epoch, test_SWA
from load_dataset import load_dataset
from selection_methods_V2 import *
from config import *
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR, CosineAnnealingWarmRestarts
from typing import List, Optional, Tuple, Dict


parser = argparse.ArgumentParser()
parser.add_argument("-l","--lambda_loss",type=float, default=1.2,
                    help="Adjustment graph loss parameter between the labeled and unlabeled")
parser.add_argument("-s","--s_margin", type=float, default=0.1,
                    help="Confidence margin of graph")
parser.add_argument("-n","--hidden_units", type=int, default=128,
                    help="Number of hidden units of the graph")
parser.add_argument("-r","--dropout_rate", type=float, default=0.3,
                    help="Dropout rate of the graph neural network")
parser.add_argument("-d","--dataset", type=str, default="cifar10",
                    help="")
parser.add_argument("-e","--no_of_epochs", type=int, default=200,
                    help="Number of epochs for the active learner")
parser.add_argument("-m","--method_type", type=str, default="Random",
                    help="")
parser.add_argument("-c","--cycles", type=int, default=10,
                    help="Number of active learning cycles")
parser.add_argument("-t","--total", type=bool, default=False,
                    help="Training on the entire dataset")
parser.add_argument("-g","--gpu-id", type=int, default=1)
parser.add_argument("--seed", type=int, default=20)
parser.add_argument("--dim_emb", type=int, default=512)

args = parser.parse_args()

config = {
    "swa_start": 100,
    "swa_anneal_epochs": 50,
    "swa_lr_multiplier": 10.0,
    "swa_scheduler_type": "constant",
    "start_swa_at_end": True,
    "optimizer_type": "sgd",
    "learning_rate": 0.1,
    "momentum": 0.9,
    "weight_decay": 0.01,
    "lr_scheduler_type": "onecycle",
    "lr_scheduler_param": 10.0,
    "snapshot_lr_multiplier":10.0,
    "snapshot_anneal_epochs":50
}
# snapshot_args.add_argument('--snapshot_start', type=int, default=100)
# snapshot_args.add_argument('--snapshot_scheduler_type', type=str, default="constant", choices=["none", "constant", "cosine"])
# snapshot_args.add_argument('--start_snapshot_at_end', action='store_true')
# snapshot_args.add_argument('--ft_epochs', type=int, default=50)
# snapshot_args.add_argument('--ft_snapshot_start', type=int, default=45)

def _query_impl(model, dataloader):
    all_probs = []
    model.eval()

    with torch.no_grad():
        for imgs, _ in dataloader:
            imgs = imgs.cuda()
            scores, _, _ = model(imgs)
            preds = F.softmax(scores, dim=1)
            all_probs.append(preds.cpu())
    all_probs = torch.cat(all_probs, dim=0)
    return all_probs.unsqueeze(1)

def postprocess(query_results):
    query_results = torch.cat(query_results, dim=1).cpu().numpy()  # [B, K, C]
    all_probs = query_results.mean(axis=1)  # [B, C]
    all_probs = np.asarray(all_probs)
    all_probs.sort(axis=-1)
    margin = all_probs[:, -1] - all_probs[:, -2]
    return margin

def Snapshot_query(_model, checkpoints, dataloader, unlabeled_set):
    model = copy.deepcopy(_model)
    query_results = []

    for ckpt in checkpoints:
        model.load_state_dict(torch.load(ckpt)["state_dict"])
        query_results.append(_query_impl(model, dataloader))
    scores = postprocess(query_results)
    query_results = scores.argsort() # [Front indices are more informative]
    query_idx = np.array(unlabeled_set)[query_results]

    return query_idx

def create_scheduler(optimizer):

    return MultiStepLR(optimizer, milestones=MILESTONES)

def create_swa_model_and_scheduler(config, model: nn.Module, optimizer: optim.Optimizer, save_interval: int) -> Tuple[AveragedModel, LambdaLR]:
    swa_model = AveragedModel(model)

    if config["swa_scheduler_type"] == "constant":
        swa_scheduler = SWALR(
            optimizer, swa_lr=config["learning_rate"]*config["snapshot_lr_multiplier"],
            anneal_epochs=config["snapshot_anneal_epochs"], anneal_strategy="cos"
        )
    elif config["swa_scheduler_type"] == "cosine":
        swa_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=save_interval, T_mult=1, eta_min=1e-5)
        swa_scheduler.base_lrs = [config.snapshot_lr_multiplier*config["learning_rate"] \
            for base_lr in swa_scheduler.base_lrs]
    elif config["swa_scheduler_tyhpe"] == "none":
        swa_scheduler = LambdaLR(optimizer, lambda epoch: 1)
    else:
        raise ValueError

    return swa_model, swa_scheduler

# Main
if __name__ == '__main__':
    args = parser.parse_args()
    
    random_seed = args.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


    if args.dataset == 'cifar100':
        args.num_cls = 100
    else:
        args.num_cls = 10
    torch.cuda.set_device(args.gpu_id)
    _device = torch.device(f"cuda:{args.gpu_id}")
    torch.set_num_threads(2)

    method = args.method_type
    methods = ['Random', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss','VAAL','TA-VAAL','ALFA-MIX','Snapshot']
    datasets = ['cifar10','cifar10im', 'cifar100', 'fashionmnist','svhn']
    assert method in methods, 'No method %s! Try options %s'%(method, methods)
    assert args.dataset in datasets, 'No dataset %s! Try options %s'%(args.dataset, datasets)
    '''
    method_type: 'Random', 'UncertainGCN', 'CoreGCN', 'CoreSet', 'lloss','VAAL','TA-VAAL','ALFA-MIX', 'Snapshot'
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
                resnet18    = resnet.ResNet18(num_classes=NO_CLASSES).cuda()

            models      = {'backbone': resnet18}
            torch.backends.cudnn.benchmark = True

            criterion      = nn.CrossEntropyLoss(reduction='none')
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            optimizers = {'backbone': optim_backbone}
            schedulers = {'backbone': sched_backbone}
            scheduler = create_scheduler(optimizers['backbone'])
            swa_model, swa_scheduler = create_swa_model_and_scheduler(config, models['backbone'], optimizers['backbone'], save_interval=20)

            train2(models, method, criterion, optimizers, schedulers, dataloaders, args.no_of_epochs, EPOCHL)
            acc = test2(models, EPOCH, method, dataloaders, mode='test')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial + 1, TRIALS, cycle + 1, CYCLES, len(labeled_set), acc))

            # Train model again for SWA selection #
            if args.dataset == "fashionmnist":
                resnet18    = resnet.ResNet18fm(num_classes=NO_CLASSES).cuda()
            else:
                resnet18    = resnet.ResNet18(num_classes=NO_CLASSES).cuda()

            models      = {'backbone': resnet18}
            torch.backends.cudnn.benchmark = True

            criterion      = nn.CrossEntropyLoss(reduction='none')
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)
            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            optimizers = {'backbone': optim_backbone}
            schedulers = {'backbone': sched_backbone}
            scheduler = create_scheduler(optimizers['backbone'])
            swa_model, swa_scheduler = create_swa_model_and_scheduler(config, models['backbone'], optimizers['backbone'], save_interval=20)

            # ============================================= COMMENT OUT ================================================ #
            print('>> Train a Model.')
            checkpoints: List[str] = []
            for epoch in tqdm(range(args.no_of_epochs)):

                loss = train_epoch(models, method, criterion, optimizers, dataloaders, epoch, EPOCHL)
                # schedulers['backbone'].step()

                if epoch > config["swa_start"]:
                    swa_model.update_parameters(models['backbone'])
                    swa_scheduler.step()

                    if epoch in [119, 139, 159, 179, 199]:
                        ckpt_file_name = os.path.join('Snapshot_Save', f"epoch_{method}_{args.dataset}_{epoch+1}_lr_{LR}.ckpt")
                        torch.save({"state_dict": models['backbone'].state_dict()}, ckpt_file_name)
                        checkpoints.append(ckpt_file_name)

            print('>> Training.')

            swa_model.cuda()
            update_bn(dataloaders['train'], swa_model, device=_device)
            swa_ckpt_name = os.path.join('Snapshot_Save', f"swa_model.ckpt")
            torch.save({"state_dict": swa_model.state_dict()}, swa_ckpt_name)

            SGD_acc = test2(models, EPOCH, method, dataloaders, mode='test')
            SWA_acc = test_SWA(swa_model, EPOCH, method, dataloaders, mode='test')
            print('Trial {}/{} || Cycle {}/{} || Label set size {}: SGD Test acc {}, SWA Test acc {}'.
                  format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), SGD_acc, SWA_acc))
            # ============================================= COMMENT OUT ================================================ #

            # Get the indices of the unlabeled samples to train on next cycle
            UL_loader = DataLoader(data_train, batch_size=BATCH, sampler=SubsetSequentialSampler(unlabeled_set), pin_memory=True)
            arg = Snapshot_query(models['backbone'], checkpoints, UL_loader, unlabeled_set)[:ADDENDUM] # Ascending
            # arg = Snapshot_query(models['backbone'], checkpoints, UL_loader, unlabeled_set)[-ADDENDUM:] # Descending
            # arg = query_samples2(models, method, data_unlabeled, subset, labeled_set, cycle, args, UL_idces=unlabeled_set, data = data_train) # ALFA-MIX

            # ============================================= COMMENT OUT ================================================ #
            # if args.dataset == "fashionmnist":
            #     models = {'backbone':resnet.ResNet18fm(num_classes=NO_CLASSES).cuda()}
            # else:
            #     models = {'backbone':resnet.ResNet18(num_classes=NO_CLASSES).cuda()}
            # optimizers = {'backbone': optim.SGD(models['backbone'].parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WDECAY)}
            # schedulers = {'backbone': lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)}
            # train2(models, method, criterion, optimizers, schedulers, dataloaders, args.no_of_epochs, EPOCHL)
            # acc = test2(models, EPOCH, method, dataloaders, mode='test')
            # print('Trial {}/{} || Cycle {}/{} || Label set size {}: Clean acc {}'.format(trial + 1, TRIALS, cycle + 1, CYCLES, len(labeled_set), acc))
            # ============================================= COMMENT OUT ================================================ #

            if cycle == (CYCLES - 1):
                # Reached final training cycle
                print("Finished.")
                break

            # Update the labeled dataset and the unlabeled dataset, respectively

            if method == "ALFA-MIX" or method == "Snapshot":
                labeled_set = list(set(labeled_set) | set(arg))
                unlabeled_set = list(set(unlabeled_set) - set(arg))
            elif method == "Random":
                arg = np.random.choice(unlabeled_set, adden, replace=False)
                labeled_set = list(set(labeled_set) | set(arg))
                unlabeled_set = list(set(unlabeled_set) - set(arg))
            else:
                labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
                listd = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy())
                unlabeled_set = listd + unlabeled_set[SUBSET:]
            # if method != "ALFA-MIX":
            #     labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
            #     listd = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy())
            #     unlabeled_set = listd + unlabeled_set[SUBSET:]
            # else:
            #     labeled_set += list(arg)
            #     unlabeled_set = list(set(unlabeled_set) - set(arg))

            print(len(labeled_set), min(labeled_set), max(labeled_set))
            dataloaders['train'] = DataLoader(data_train, batch_size=BATCH, sampler=SubsetRandomSampler(labeled_set), pin_memory=True)

    results.close()