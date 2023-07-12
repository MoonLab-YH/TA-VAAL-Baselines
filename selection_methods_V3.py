import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
# Custom
from config import *
from models.query_models import VAE, Discriminator, GCN
from data.sampler import SubsetSequentialSampler
from kcenterGreedy import kCenterGreedy
from tqdm import tqdm
from collections import OrderedDict


def BCEAdjLoss(scores, lbl, nlbl, l_adj):
    lnl = torch.log(scores[lbl])
    lnu = torch.log(1 - scores[nlbl])
    labeled_score = torch.mean(lnl)
    unlabeled_score = torch.mean(lnu)
    bce_adj_loss = -labeled_score - l_adj * unlabeled_score
    return bce_adj_loss


def aff_to_adj(x, y=None):
    x = x.detach().cpu().numpy()
    adj = np.matmul(x, x.transpose())
    adj += -1.0 * np.eye(adj.shape[0])
    adj_diag = np.sum(adj, axis=0)  # rowise sum
    adj = np.matmul(adj, np.diag(1 / adj_diag))
    adj = adj + np.eye(adj.shape[0])
    adj = torch.Tensor(adj).cuda()

    return adj


def read_data(dataloader, labels=True):
    if labels:
        while True:
            for img, label, _ in dataloader:
                yield img, label
    else:
        while True:
            for img, _, _ in dataloader:
                yield img


def vae_loss(x, recon, mu, logvar, beta):
    mse_loss = nn.MSELoss()
    MSE = mse_loss(recon, x)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD * beta
    return MSE + KLD


def train_vaal(models, optimizers, labeled_dataloader, unlabeled_dataloader, cycle):
    vae = models['vae']
    discriminator = models['discriminator']
    task_model = models['backbone']
    ranker = models['module']

    task_model.eval()
    ranker.eval()
    vae.train()
    discriminator.train()
    vae = vae.cuda()
    discriminator = discriminator.cuda()
    task_model = task_model.cuda()
    ranker = ranker.cuda()
    adversary_param = 1
    beta = 1
    num_adv_steps = 1
    num_vae_steps = 1

    bce_loss = nn.BCELoss()

    labeled_data = read_data(labeled_dataloader)
    unlabeled_data = read_data(unlabeled_dataloader)

    train_iterations = int((ADDENDUM * cycle + SUBSET) * EPOCHV / BATCH)

    with tqdm(total=train_iterations) as pbar:
        for iter_count in range(train_iterations):
            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)[0]
            lossDict = {'predLoss': [], 'L_vaeLoss': [], 'UL_vaeLoss': [], 'disLoss': []}

            labeled_imgs = labeled_imgs.cuda()
            unlabeled_imgs = unlabeled_imgs.cuda()
            labels = labels.cuda()
            if iter_count == 0:
                r_l_0 = torch.from_numpy(np.random.uniform(0, 1, size=(labeled_imgs.shape[0], 1))).type(
                    torch.FloatTensor).cuda()
                r_u_0 = torch.from_numpy(np.random.uniform(0, 1, size=(unlabeled_imgs.shape[0], 1))).type(
                    torch.FloatTensor).cuda()
            else:
                with torch.no_grad():
                    _, _, features_l = task_model(labeled_imgs)
                    _, _, feature_u = task_model(unlabeled_imgs)
                    r_l = ranker(features_l)
                    r_u = ranker(feature_u)
            if iter_count == 0:
                r_l = r_l_0.detach()
                r_u = r_u_0.detach()
                r_l_s = r_l_0.detach()
                r_u_s = r_u_0.detach()
            else:
                r_l_s = torch.sigmoid(r_l).detach()
                r_u_s = torch.sigmoid(r_u).detach()
            # VAE step
            for count in range(num_vae_steps):  # num_vae_steps
                recon, _, mu, logvar = vae(r_l_s, labeled_imgs)
                unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, beta)
                unlab_recon, _, unlab_mu, unlab_logvar = vae(r_u_s, unlabeled_imgs)
                transductive_loss = vae_loss(unlabeled_imgs, unlab_recon, unlab_mu, unlab_logvar, beta)

                labeled_preds = discriminator(r_l, mu)
                unlabeled_preds = discriminator(r_u, unlab_mu)

                lab_real_preds = torch.ones(labeled_imgs.size(0))
                unlab_real_preds = torch.ones(unlabeled_imgs.size(0))

                lab_real_preds = lab_real_preds.cuda()
                unlab_real_preds = unlab_real_preds.cuda()

                dsc_loss = bce_loss(labeled_preds[:, 0], lab_real_preds) + bce_loss(unlabeled_preds[:, 0], unlab_real_preds)
                total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss

                optimizers['vae'].zero_grad()
                total_vae_loss.backward()
                optimizers['vae'].step()

                # sample new batch if needed to train the adversarial network
                if count < (num_vae_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)[0]

                    labeled_imgs = labeled_imgs.cuda()
                    unlabeled_imgs = unlabeled_imgs.cuda()
                    labels = labels.cuda()

            # Discriminator step
            for count in range(num_adv_steps):
                with torch.no_grad():
                    _, _, mu, _ = vae(r_l_s, labeled_imgs)
                    _, _, unlab_mu, _ = vae(r_u_s, unlabeled_imgs)

                labeled_preds = discriminator(r_l, mu)
                unlabeled_preds = discriminator(r_u, unlab_mu)

                lab_real_preds = torch.ones(labeled_imgs.size(0))
                unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

                lab_real_preds = lab_real_preds.cuda()
                unlab_fake_preds = unlab_fake_preds.cuda()

                dsc_loss = bce_loss(labeled_preds[:, 0], lab_real_preds) + \
                           bce_loss(unlabeled_preds[:, 0], unlab_fake_preds)

                optimizers['discriminator'].zero_grad()
                dsc_loss.backward()
                optimizers['discriminator'].step()

                # sample new batch if needed to train the adversarial network
                if count < (num_adv_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)[0]

                    labeled_imgs = labeled_imgs.cuda()
                    unlabeled_imgs = unlabeled_imgs.cuda()
                    labels = labels.cuda()
                if iter_count % 100 == 0:
                    print("Iteration: " + str(iter_count) + "  vae_loss: " + str(
                        total_vae_loss.item()) + " dsc_loss: " + str(dsc_loss.item()))

            lossDict['L_vaeLoss'] = unsup_loss.item()
            lossDict['UL_vaeLoss'] = transductive_loss.item()
            lossDict['disLoss'] = dsc_loss.item()
            lossPostfix = OrderedDict({k: torch.tensor(v).float().mean() for (k, v) in lossDict.items()})
            disPostfix = OrderedDict({'lb_dist': '{0:.4f}'.format(labeled_preds.mean().item()),
                                      'unlb_dist': '{0:.4f}'.format(unlabeled_preds.mean().item())})
            postfix = OrderedDict(list(lossPostfix.items()) + list(disPostfix.items()))
            pbar.set_postfix(**postfix)
            pbar.update(1)


def get_uncertainty(models, unlabeled_loader):
    models['backbone'].eval()
    models['module'].eval()
    uncertainty = torch.tensor([]).cuda()

    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            inputs = inputs.cuda()
            _, _, features = models['backbone'](inputs)
            pred_loss = models['module'](features)  # pred_loss = criterion(scores, labels) # ground truth loss
            pred_loss = pred_loss.view(pred_loss.size(0))
            uncertainty = torch.cat((uncertainty, pred_loss), 0)

    return uncertainty.cpu()


def get_features(models, unlabeled_loader):
    models['backbone'].eval()
    features = torch.tensor([]).cuda()
    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            inputs = inputs.cuda()
            _, features_batch, _ = models['backbone'](inputs)
            features = torch.cat((features, features_batch), 0)
        feat = features  # .detach().cpu().numpy()
    return feat


def get_kcg(models, labeled_data_size, unlabeled_loader):
    models['backbone'].eval()
    features = torch.tensor([]).cuda()

    with torch.no_grad():
        for inputs, _, _ in unlabeled_loader:
            inputs = inputs.cuda()
            _, features_batch, _ = models['backbone'](inputs)
            features = torch.cat((features, features_batch), 0)
        feat = features.detach().cpu().numpy()
        new_av_idx = np.arange(SUBSET, (SUBSET + labeled_data_size))
        sampling = kCenterGreedy(feat)
        batch = sampling.select_batch_(new_av_idx, ADDENDUM)
        other_idx = [x for x in range(SUBSET) if x not in batch]
    return other_idx + batch


# Select the indices of the unlablled data according to the methods
def query_samples3(model, method, data_unlabeled, subset, labeled_set, cycle, args):
    if method == 'Random':
        arg = np.random.randint(SUBSET, size=SUBSET)

    if method == 'TA-VAAL':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, sampler=SubsetSequentialSampler(subset),
                                      pin_memory=True)
        labeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, sampler=SubsetSequentialSampler(labeled_set),
                                    pin_memory=True)
        if args.dataset == 'fashionmnist':
            vae = VAE(28, 1, 3)
            discriminator = Discriminator(28)
        else:
            vae = VAE()
            discriminator = Discriminator(32)

        models = {'backbone': model['backbone'], 'module': model['module'], 'vae': vae, 'discriminator': discriminator}

        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
        optimizers = {'vae': optim_vae, 'discriminator': optim_discriminator}

        train_vaal(models, optimizers, labeled_loader, unlabeled_loader, cycle + 1)
        task_model = models['backbone']
        ranker = models['module']
        all_preds, all_indices = [], []

        for images, _, indices in unlabeled_loader:
            images = images.cuda()
            with torch.no_grad():
                _, _, features = task_model(images)
                r = ranker(features)
                _, _, mu, _ = vae(torch.sigmoid(r), images)
                preds = discriminator(r, mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk
        all_preds *= -1
        # select the points which the discriminator things are the most likely to be unlabeled
        _, arg = torch.sort(all_preds)

    arg = np.random.randint(SUBSET, size=SUBSET)
    print(f'mean of arg is {arg.mean()}')
    return arg
