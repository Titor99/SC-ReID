import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import shutil
from ignite.metrics import Metric
import torch.nn.functional as F
from PIL import Image, ImageOps

def imshow(path, title=None, border=None):
    """Imshow for Tensor."""
    # im = plt.imread(path)
    im = Image.open(path)
    im1 = im.resize((80, 160))
    if border is not None:
        im2 = ImageOps.expand(im1, border=5, fill=border)
        plt.imshow(im2)
    else:
        plt.imshow(im1)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):     # [3368, 15913], [3368,], [15913,]
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)   # [3368, 15913]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)    # [3368, 15913]

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):       # 3368
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)      # [15913,]
        keep = np.invert(remove)      # [15913,]

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]       # [15908,]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()          # [15908,]
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()         # 14
        tmp_cmc = orig_cmc.cumsum()      # [15908,], [0,0,0,...,14,14,14]
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]     # [15908,]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc    # [15908,]
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q      # [50,]
    mAP = np.mean(all_AP)

    del indices, matches

    return all_cmc, mAP

class R1_mAP(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reset()

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)       # [19281, 2048]
        if self.feat_norm == 'yes':
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        qf = feats[:self.num_query]                           # [3368, 2048]
        q_pids = np.asarray(self.pids[:self.num_query])       # [3368,]
        q_camids = np.asarray(self.camids[:self.num_query])   # [3368,]
        # gallery
        gf = feats[self.num_query:]                           # [15913, 2048]
        g_pids = np.asarray(self.pids[self.num_query:])       # [15913,]
        g_camids = np.asarray(self.camids[self.num_query:])   # [15913,]
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())                     # [3368, 15913]
        distmat = distmat - 2 * torch.matmul(qf, gf.t())
        distmat = distmat.cpu().numpy()
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP

def fliplr(img):
    # flip horizontal
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.cpu().index_select(3, inv_idx)
    return img_flip.cuda()

def inference_prcc_global(model, test_loader, num_query, use_flip=True):
    print('Test')
    model.eval()
    metric = R1_mAP(num_query, 500)
    with torch.no_grad():
        for ii, batch in enumerate(test_loader):
            data, pid, path, camid = batch                          # [128, 3, 256, 128]
            data = data.to("cuda") if torch.cuda.device_count() >= 1 else data
            b, c, h, w = data.shape

            feat = model(data)

            if use_flip:
                data_f = fliplr(data.cpu()).cuda()              # [128, 3, 256, 128]
                feat_f = model(data_f)
                feat += feat_f

            feat = F.normalize(feat, p=2, dim=-1)

            metric.update([feat, pid, camid])
    cmc, mAP = metric.compute()
    return mAP, cmc[0]



def inference_prcc_visual_rank(model, test_loader, num_query, home, show_rank=20, use_flip=True, max_rank=50):
    print('Visualize')

    if not os.path.exists(home):
        os.makedirs(home)

    model.eval()
    feats = []
    pids = []
    camids = []
    fnames_all = []
    num_total = len(test_loader)
    with torch.no_grad():
        for ii, batch in enumerate(test_loader):
            # data, pid, cmp, fnames, mask = batch                          # [128, 3, 256, 128]
            data, pid, fnames, mask, face = batch
            data = data.to("cuda") if torch.cuda.device_count() >= 1 else data
            b, c, h, w = data.shape

            feat = model(data)                              # [64, 4*2048]

            if use_flip:
                data_f = fliplr(data.cpu()).cuda()              # [128, 3, 256, 128]
                feat_f = model(data_f)
                feat += feat_f
            feat = F.normalize(feat, p=2, dim=-1)

            feats.append(feat)
            pids.extend(pid)
            camids.extend(fnames)
            fnames_all.extend(fnames)

            if ii % 100 == 0:
                print(ii+1, '/', num_total)



    feats = torch.cat(feats, dim=0)  # [6927, 2048]
    feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # [6927, 2048]
    pids = np.array(pids)
    camids = np.array(camids)
    fnames_all = np.array(fnames_all)

    # query
    qf = feats[:num_query]  # [3368, 2048]
    q_pids = np.asarray(pids[:num_query])  # [3368,]
    q_camids = np.asarray(camids[:num_query])  # [3368,]
    q_fnames = fnames_all[:num_query]
    # gallery
    gf = feats[num_query:]  # [15913, 2048]
    g_pids = np.asarray(pids[num_query:])  # [15913,]
    g_camids = np.asarray(camids[num_query:])  # [15913,]
    g_fnames = fnames_all[num_query:]

    m, n = qf.shape[0], gf.shape[0]
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat = distmat - 2 * torch.matmul(qf, gf.t())
    distmat = distmat.cpu().numpy()

    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
    indices = np.argsort(distmat, axis=1)  # [3368, 15913]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)  # [3368, 15913]

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    print('num_q:', num_q)
    for q_idx in range(num_q):  # 3368
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        q_fname = q_fnames[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)  # [15913,]
        keep = np.invert(remove)  # [15913,]

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]  # [15908,]

        if not np.any(orig_cmc):
            continue

        g_fnames_s = g_fnames[order][keep][0: show_rank]
        g_pids_s = g_pids[order][keep][0: show_rank]

        # copy query
        src = q_fname
        q_fname_list = q_fname.split('/')
        name_dir = q_fname_list[-3] + '_' + q_fname_list[-2] + '_' + q_fname_list[-1]
        path_d = os.path.join(home, name_dir)
        if not os.path.exists(path_d):
            os.makedirs(path_d)
        dst = os.path.join(path_d, name_dir)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)

        # copy gallery
        path_g = os.path.join(path_d, 'gallery')
        if not os.path.exists(path_g):
            os.makedirs(path_g)

        for kk, g_fname in enumerate(g_fnames_s):
            src = g_fname
            g_fname_list = g_fname.split('/')
            if q_pid == g_pids_s[kk]:
                name_dir = str(kk + 1).zfill(3) + '_' + 'right' + '_' + g_fname_list[-3] + '_' + g_fname_list[-2] + '_' + g_fname_list[-1]
            else:
                name_dir = str(kk + 1).zfill(3) + '_' + 'wrong' + '_' + g_fname_list[-3] + '_' + g_fname_list[-2] + '_' + g_fname_list[-1]

            dst = os.path.join(path_g, name_dir)
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)

def inference_prcc_visual_rank_easy(model, test_loader, num_query, home, show_rank=20, use_flip=True, max_rank=50):
    print('Easy Visualize')

    home = os.path.join(home, 'easy')
    if not os.path.exists(home):
        os.makedirs(home)

    model.eval()
    feats = []
    pids = []
    camids = []
    fnames_all = []
    num_total = len(test_loader)
    with torch.no_grad():
        for ii, batch in enumerate(test_loader):
            # data, pid, cmp, fnames, mask = batch                          # [128, 3, 256, 128]
            data, pid, fnames, mask, face = batch
            data = data.to("cuda") if torch.cuda.device_count() >= 1 else data
            b, c, h, w = data.shape

            feat = model(data)                              # [64, 4*2048]

            if use_flip:
                data_f = fliplr(data.cpu()).cuda()              # [128, 3, 256, 128]
                feat_f = model(data_f)
                feat += feat_f
            feat = F.normalize(feat, p=2, dim=-1)

            feats.append(feat)
            pids.extend(pid)
            camids.extend(fnames)
            fnames_all.extend(fnames)

            if ii % 100 == 0:
                print(ii+1, '/', num_total)


    feats = torch.cat(feats, dim=0)  # [6927, 2048]
    feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # [6927, 2048]
    pids = np.array(pids)
    camids = np.array(camids)
    fnames_all = np.array(fnames_all)

    # query
    qf = feats[:num_query]  # [3368, 2048]
    q_pids = np.asarray(pids[:num_query])  # [3368,]
    q_camids = np.asarray(camids[:num_query])  # [3368,]
    q_fnames = fnames_all[:num_query]
    # gallery
    gf = feats[num_query:]  # [15913, 2048]
    g_pids = np.asarray(pids[num_query:])  # [15913,]
    g_camids = np.asarray(camids[num_query:])  # [15913,]
    g_fnames = fnames_all[num_query:]

    m, n = qf.shape[0], gf.shape[0]
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat = distmat - 2 * torch.matmul(qf, gf.t())
    distmat = distmat.cpu().numpy()

    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
    indices = np.argsort(distmat, axis=1)  # [3368, 15913]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)  # [3368, 15913]

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    print('num_q:', num_q)
    for q_idx in range(num_q):  # 3368

        result = Image.new('RGB', (1600,400), color=(255, 255, 255))
        name = str(q_idx) + '.png'
        save_path = os.path.join(home, name)
        result.save(save_path)
        result.close()

        fig = plt.figure(figsize=(16, 4))
        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        q_fname = q_fnames[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)  # [15913,]
        keep = np.invert(remove)  # [15913,]

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]  # [15908,]

        if not np.any(orig_cmc):
            continue

        g_fnames_s = g_fnames[order][keep][0: show_rank]
        g_pids_s = g_pids[order][keep][0: show_rank]

        # copy query
        src = q_fname
        imshow(src, 'query')

        for kk, g_fname in enumerate(g_fnames_s):

            ax = plt.subplot(1, 11, kk + 2)
            ax.axis('off')
            src = g_fname

            print(q_pid, g_pids_s[kk])
            if q_pid == g_pids_s[kk]:
                ax.set_title('%d' % (kk + 1), color='green')
                imshow(src, border='green')
            else:
                ax.set_title('%d' % (kk + 1), color='red')
                imshow(src, border='red')


        fig.savefig(save_path)
        plt.close(fig)
