import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
# Custom
from config import *
from models.query_models import VAE, Discriminator, GCN
from data.sampler import SubsetSequentialSampler
from kcenterGreedy import kCenterGreedy
from tqdm import tqdm
from collections import OrderedDict
import torch.nn.functional as F
from load_dataset_ALFAMIX import *
import torchvision.transforms as T
import copy
from sklearn.cluster import KMeans

def sample(n, feats):
    feats = feats.numpy()
    cluster_learner = KMeans(n_clusters=n)
    cluster_learner.fit(feats)

    cluster_idxs = cluster_learner.predict(feats)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (feats - centers) ** 2
    dis = dis.sum(axis=1)
    return np.array(
        [np.arange(feats.shape[0])[cluster_idxs == i][dis[cluster_idxs == i].argmin()] for i in range(n) if
         (cluster_idxs == i).sum() > 0])


def generate_alpha(size, embedding_size, alpha_cap):
    alpha = torch.normal(
        mean=alpha_cap / 2.0,
        std=alpha_cap / 2.0,
        size=(size, embedding_size))

    alpha[torch.isnan(alpha)] = 1
    return torch.clamp(alpha, min=1e-8, max=alpha_cap)

def find_candidate_set(lb_embedding, ulb_embedding, pred_1, ulb_probs, alpha_cap, Y, grads, args, model):
    unlabeled_size = ulb_embedding.size(0)
    embedding_size = ulb_embedding.size(1)

    min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float)
    pred_change = torch.zeros(unlabeled_size, dtype=torch.bool)

    for i in range(args.num_cls):
        emb = lb_embedding[Y == i]
        if emb.size(0) == 0:
            emb = lb_embedding
        anchor_i = emb.mean(dim=0).view(1, -1).repeat(unlabeled_size, 1)

        alpha = generate_alpha(unlabeled_size, embedding_size, alpha_cap)
        embedding_mix = (1 - alpha) * ulb_embedding + alpha * anchor_i
        out = model.linear(embedding_mix.cuda())
        # out, _ = model.clf(embedding_mix.cuda(), embedding=True)
        out = out.detach().cpu()
        pc = out.argmax(dim=1) != pred_1
        torch.cuda.empty_cache()

        alpha[~pc] = 1.
        pred_change[pc] = True
        is_min = min_alphas.norm(dim=1) > alpha.norm(dim=1)
        min_alphas[is_min] = alpha[is_min]

    return pred_change, min_alphas

def BCEAdjLoss(scores, lbl, nlbl, l_adj):
    lnl = torch.log(scores[lbl])
    lnu = torch.log(1 - scores[nlbl])
    labeled_score = torch.mean(lnl) 
    unlabeled_score = torch.mean(lnu)
    bce_adj_loss = -labeled_score - l_adj*unlabeled_score
    return bce_adj_loss


def aff_to_adj(x, y=None):
    x = x.detach().cpu().numpy()
    adj = np.matmul(x, x.transpose())
    adj +=  -1.0*np.eye(adj.shape[0])
    adj_diag = np.sum(adj, axis=0) #rowise sum
    adj = np.matmul(adj, np.diag(1/adj_diag))
    adj = adj + np.eye(adj.shape[0])
    adj = torch.Tensor(adj).cuda()

    return adj

def read_data(dataloader, labels=True):
    if labels:
        while True:
            for img, label,_ in dataloader:
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
    beta          = 1
    num_adv_steps = 1
    num_vae_steps = 1

    bce_loss = nn.BCELoss()
    
    labeled_data = read_data(labeled_dataloader)
    unlabeled_data = read_data(unlabeled_dataloader)

    train_iterations = int( (ADDENDUM*cycle+ SUBSET) * EPOCHV / BATCH )

    with tqdm(total=train_iterations) as pbar:
        for iter_count in range(train_iterations):
            labeled_imgs, labels = next(labeled_data)
            unlabeled_imgs = next(unlabeled_data)[0]
            lossDict = {'predLoss': [], 'L_vaeLoss': [], 'UL_vaeLoss': [], 'disLoss': []}

            labeled_imgs = labeled_imgs.cuda()
            unlabeled_imgs = unlabeled_imgs.cuda()
            labels = labels.cuda()
            if iter_count == 0 :
                r_l_0 = torch.from_numpy(np.random.uniform(0, 1, size=(labeled_imgs.shape[0],1))).type(torch.FloatTensor).cuda()
                r_u_0 = torch.from_numpy(np.random.uniform(0, 1, size=(unlabeled_imgs.shape[0],1))).type(torch.FloatTensor).cuda()
            else:
                with torch.no_grad():
                    _,_,features_l = task_model(labeled_imgs)
                    _,_,feature_u = task_model(unlabeled_imgs)
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
            for count in range(num_vae_steps): # num_vae_steps
                recon, _, mu, logvar = vae(r_l_s, labeled_imgs)
                unsup_loss = vae_loss(labeled_imgs, recon, mu, logvar, beta)
                unlab_recon, _, unlab_mu, unlab_logvar = vae(r_u_s, unlabeled_imgs)
                transductive_loss = vae_loss(unlabeled_imgs, unlab_recon, unlab_mu, unlab_logvar, beta)

                labeled_preds = discriminator(r_l, mu)
                unlabeled_preds = discriminator(r_u, unlab_mu)

                lab_real_preds = torch.ones(labeled_imgs.size(0))
                unlab_real_preds = torch.ones(unlabeled_imgs.size(0))

                lab_real_preds = lab_real_preds.cuda() # 1
                unlab_real_preds = unlab_real_preds.cuda() # 1

                dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + bce_loss(unlabeled_preds[:,0], unlab_real_preds)
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
                    _, _, mu, _ = vae(r_l_s,labeled_imgs)
                    _, _, unlab_mu, _ = vae(r_u_s,unlabeled_imgs)

                labeled_preds = discriminator(r_l, mu)
                unlabeled_preds = discriminator(r_u, unlab_mu)

                lab_real_preds = torch.ones(labeled_imgs.size(0)) # 1
                unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0)) # 0

                lab_real_preds = lab_real_preds.cuda()
                unlab_fake_preds = unlab_fake_preds.cuda()

                dsc_loss = bce_loss(labeled_preds[:,0], lab_real_preds) + bce_loss(unlabeled_preds[:,0], unlab_fake_preds)

                optimizers['discriminator'].zero_grad()
                dsc_loss.backward()
                optimizers['discriminator'].step()

                # sample new batch if needed to train the adversarial network
                if count < (num_adv_steps-1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)[0]

                    labeled_imgs = labeled_imgs.cuda()
                    unlabeled_imgs = unlabeled_imgs.cuda()
                    labels = labels.cuda()
                if iter_count % 100 == 0:
                    print("Iteration: " + str(iter_count) + "  vae_loss: " + str(total_vae_loss.item()) + " dsc_loss: " +str(dsc_loss.item()))

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
            pred_loss = models['module'](features) # pred_loss = criterion(scores, labels) # ground truth loss
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
            feat = features #.detach().cpu().numpy()
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
        new_av_idx = np.arange(SUBSET,(SUBSET + labeled_data_size))
        sampling = kCenterGreedy(feat)  
        batch = sampling.select_batch_(new_av_idx, ADDENDUM)
        other_idx = [x for x in range(SUBSET) if x not in batch]
    return  other_idx + batch


# Select the indices of the unlablled data according to the methods
def query_samples2(model, method, data_unlabeled, subset, labeled_set, cycle, args, **kwargs):

    if args.dataset == 'cifar100' or args.dataset == 'cifar100im':
        ADDENDUM = 2000

    if method == 'Random':
        arg = np.random.randint(len(kwargs['UL_idces']), size=len(kwargs['UL_idces']))

    if method == 'TA-VAAL':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, sampler=SubsetSequentialSampler(subset), pin_memory=True)
        labeled_loader = DataLoader(data_unlabeled, batch_size=BATCH, sampler=SubsetSequentialSampler(labeled_set), pin_memory=True)
        if args.dataset == 'fashionmnist':
            vae = VAE(28,1,3).cuda()
            discriminator = Discriminator(28).cuda()
        else:
            vae = VAE().cuda()
            discriminator = Discriminator(32).cuda()
     
        models      = {'backbone': model['backbone'], 'module': model['module'], 'vae': vae, 'discriminator': discriminator}
        
        optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
        optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
        optimizers = {'vae': optim_vae, 'discriminator':optim_discriminator}

        train_vaal(models, optimizers, labeled_loader, unlabeled_loader, cycle+1)
        task_model = models['backbone']
        ranker = models['module']
        all_preds, all_indices = [], []

        for images, _, indices in unlabeled_loader:
            images = images.cuda()
            with torch.no_grad():
                _,_,features = task_model(images)
                r = ranker(features)
                _, _, mu, _ = vae(torch.sigmoid(r),images)
                preds = discriminator(r,mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1
        # select the points which the discriminator things are the most likely to be unlabeled
        _, arg = torch.sort(all_preds)

    if method == 'CoreSet':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                    sampler=SubsetSequentialSampler(subset+labeled_set), # more convenient if we maintain the order of subset
                                    pin_memory=True)

        arg = get_kcg(model, ADDENDUM*(cycle+1), unlabeled_loader)

    if method == 'lloss':
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                    sampler=SubsetSequentialSampler(subset),
                                    pin_memory=True)

        # Measure uncertainty of each data points in the subset
        uncertainty = get_uncertainty(model, unlabeled_loader)
        arg = np.argsort(uncertainty)

    if (method == 'UncertainGCN') or (method == 'CoreGCN'):
        # Create unlabeled dataloader for the unlabeled subset
        unlabeled_loader = DataLoader(data_unlabeled, batch_size=BATCH,
                                      sampler=SubsetSequentialSampler(subset + labeled_set),
                                      # more convenient if we maintain the order of subset
                                      pin_memory=True)
        binary_labels = torch.cat((torch.zeros([SUBSET, 1]), (torch.ones([len(labeled_set), 1]))), 0)

        features = get_features(model, unlabeled_loader)
        features = nn.functional.normalize(features)
        adj = aff_to_adj(features)

        gcn_module = GCN(nfeat=features.shape[1],
                         nhid=args.hidden_units,
                         nclass=1,
                         dropout=args.dropout_rate).cuda()

        models = {'gcn_module': gcn_module}

        optim_backbone = optim.Adam(models['gcn_module'].parameters(), lr=LR_GCN, weight_decay=WDECAY)
        optimizers = {'gcn_module': optim_backbone}

        lbl = np.arange(SUBSET, SUBSET + (cycle + 1) * ADDENDUM, 1)
        nlbl = np.arange(0, SUBSET, 1)

        ############
        for _ in range(200):
            optimizers['gcn_module'].zero_grad()
            outputs, _, _ = models['gcn_module'](features, adj)
            lamda = args.lambda_loss
            loss = BCEAdjLoss(outputs, lbl, nlbl, lamda)
            loss.backward()
            optimizers['gcn_module'].step()

        models['gcn_module'].eval()
        with torch.no_grad():
            # with torch.cuda.device(CUDA_VISIBLE_DEVICES):
            #     inputs = features.cuda()
            #     labels = binary_labels.cuda()
            inputs = features.cuda()
            labels = binary_labels.cuda()
            scores, _, feat = models['gcn_module'](inputs, adj)

            if method == "CoreGCN":
                feat = features.detach().cpu().numpy()
                new_av_idx = np.arange(SUBSET, (SUBSET + (cycle + 1) * ADDENDUM))
                sampling2 = kCenterGreedy(feat)
                batch2 = sampling2.select_batch_(new_av_idx, ADDENDUM)
                other_idx = [x for x in range(SUBSET) if x not in batch2]
                arg = other_idx + batch2
            else:
                s_margin = args.s_margin
                scores_median = np.squeeze(torch.abs(scores[:SUBSET] - s_margin).detach().cpu().numpy())
                arg = np.argsort(-(scores_median))

            print("Max confidence value: ", torch.max(scores.data))
            print("Mean confidence value: ", torch.mean(scores.data))
            preds = torch.round(scores)
            correct_labeled = (preds[SUBSET:, 0] == labels[SUBSET:, 0]).sum().item() / ((cycle + 1) * ADDENDUM)
            correct_unlabeled = (preds[:SUBSET, 0] == labels[:SUBSET, 0]).sum().item() / SUBSET
            correct = (preds[:, 0] == labels[:, 0]).sum().item() / (SUBSET + (cycle + 1) * ADDENDUM)
            print("Labeled classified: ", correct_labeled)
            print("Unlabeled classified: ", correct_unlabeled)
            print("Total classified: ", correct)

    if method == "ALFA-MIX":
        if args.dataset == 'cifar100':
            ADDENDUM = 2000
        else:
            ADDENDUM = 1000
        UL_idces = kwargs['UL_idces']
        if args.dataset == 'fashionmnist':
            test_transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.1307], [0.3081])
            ])
        else:
            test_transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])

        handler = get_handler(args.dataset)
        # Collect Prob & Embedding for Unlabeled Set #
        X_ul = kwargs['data'].data[UL_idces]
        if args.dataset == 'svhn':
            Y_ul = np.array(kwargs['data'].labels)[UL_idces]
        else:
            Y_ul = np.array(kwargs['data'].targets)[UL_idces]
        loader_ul = DataLoader(handler(X_ul, Y_ul, transform=test_transform), batch_size=BATCH, pin_memory=True)
        ulb_probs = torch.zeros([len(UL_idces), args.num_cls])
        org_ulb_embedding = torch.zeros([len(UL_idces), args.dim_emb])
        model['backbone'].eval()
        with torch.no_grad():
            for x, y, idxs in loader_ul:
                x, y = x.cuda(), y.cuda()
                out, e1, _ = model['backbone'](x)
                prob = F.softmax(out, dim=1)
                ulb_probs[idxs] = prob.cpu()
                org_ulb_embedding[idxs] = e1.cpu()
        probs_sorted, probs_sort_idxs = ulb_probs.sort(descending=True)
        pred_1 = probs_sort_idxs[:, 0]

        # Collect Prob & Embedding for Labeled Set #
        X_l = kwargs['data'].data[labeled_set]
        if args.dataset == 'svhn':
            Y_l = np.array(kwargs['data'].labels)[labeled_set]
        else:
            Y_l = np.array(kwargs['data'].targets)[labeled_set]
        loader_l = DataLoader(handler(X_l, Y_l, transform=test_transform), batch_size=BATCH, pin_memory=True)
        lb_probs = torch.zeros([len(labeled_set), args.num_cls])
        org_lb_embedding = torch.zeros([len(labeled_set), args.dim_emb])
        model['backbone'].eval()
        with torch.no_grad():
            for x, y, idxs in loader_l:
                x, y = x.cuda(), y.cuda()
                out, e1, _ = model['backbone'](x)
                prob = F.softmax(out, dim=1)
                lb_probs[idxs] = prob.cpu()
                org_lb_embedding[idxs] = e1.cpu()

        ulb_embedding = org_ulb_embedding
        lb_embedding = org_lb_embedding

        unlabeled_size = ulb_embedding.size(0)
        embedding_size = ulb_embedding.size(1)

        min_alphas = torch.ones((unlabeled_size, embedding_size), dtype=torch.float)
        candidate = torch.zeros(unlabeled_size, dtype=torch.bool)

        grads = None

        alpha_cap = 0.
        while alpha_cap < 1.0:
            alpha_cap += 0.3125 # self.args.alpha_cap
            # alpha_cap += 0.03125 # self.args.alpha_cap

            tmp_pred_change, tmp_min_alphas = find_candidate_set(lb_embedding, ulb_embedding, pred_1, ulb_probs,
                                    alpha_cap=alpha_cap, Y=Y_l, grads=grads, args=args, model = model['backbone'])

            is_changed = min_alphas.norm(dim=1) >= tmp_min_alphas.norm(dim=1)

            min_alphas[is_changed] = tmp_min_alphas[is_changed]
            candidate += tmp_pred_change

            print('With alpha_cap set to %f, number of inconsistencies: %d' % (
            alpha_cap, int(tmp_pred_change.sum().item())))

            if candidate.sum() > ADDENDUM:
                break

        if candidate.sum() > 0:
            print('Number of inconsistencies: %d' % (int(candidate.sum().item())))

            print('alpha_mean_mean: %f' % min_alphas[candidate].mean(dim=1).mean().item())
            print('alpha_std_mean: %f' % min_alphas[candidate].mean(dim=1).std().item())
            print('alpha_mean_std %f' % min_alphas[candidate].std(dim=1).mean().item())

            c_alpha = F.normalize(org_ulb_embedding[candidate].view(candidate.sum(), -1), p=2, dim=1).detach()

            selected_idxs = sample(min(ADDENDUM, candidate.sum().item()), feats=c_alpha)
            selected_idxs = np.array(UL_idces)[candidate][selected_idxs]
        else:
            selected_idxs = np.array([], dtype=np.int)

        if len(selected_idxs) < ADDENDUM:
            remained = ADDENDUM - len(selected_idxs)
            idx_lb = np.array([False]*(len(labeled_set) + len(UL_idces)))
            idx_lb[selected_idxs] = True
            selected_idxs = np.concatenate([selected_idxs, np.random.choice(np.where(idx_lb == 0)[0], remained)])
            print('picked %d samples from RandomSampling.' % (remained))

        arg = np.array(selected_idxs)

    # np.save(f'sampleIdx_{cycle}cycle.np', np.array(arg))
    # print(f'mean of arg is {arg.mean()}')
    return arg
