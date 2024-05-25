import torch
from torch import nn

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def euclidean_dist(x, y):     # [64, 2048], [64, 2048]
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)          # 64, 64
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # dist.addmm_(1, -2, x, y.t())
    dist = dist - 2 * torch.mm(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability        [64, 64]
    return dist

def hard_example_mining(dist_mat, labels, return_inds=False):        # [64, 64],  [64,]
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)          # 64

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())          # [64, 64]
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())          # [64, 64]

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)      # [64, 1],  [64, 1]
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)      # [64, 1],  [64, 1]
    # shape [N]
    dist_ap = dist_ap.squeeze(1)        # [64,]
    dist_an = dist_an.squeeze(1)        # [64,]

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels).copy_(torch.arange(0, N).long()).unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.crossentropy_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        loss = self.crossentropy_loss(inputs, targets)
        return loss

class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=0.3):
        self.margin = margin       # 0.3
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):       # [64, 2048],  [64,]
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)         # [64, 64]
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)    # [64,], [64,]
        y = dist_an.new().resize_as_(dist_an).fill_(1)              # [64,]
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):   # [64, 751],   [64,]
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)  # [64, 751]
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)         # [64, 751]   one_hot
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class MSELoss(nn.Module):
    def __init__(self, ):
        super(MSELoss, self).__init__()

    def forward(self, feat1, feat2):    # [64, 2048], [64, 2048]
        B, C = feat1.shape

        dist = torch.pow(torch.abs(feat1 - feat2), 2).sum(dim=-1)

        # loss = dist.mean()  mse
        # loss = (dist.mean()) ** 0.5  rmse
        loss = (1. / (1. + torch.exp(-dist))).mean()   # *mse

        return loss

class TripletWithFaceLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=0.3):
        self.margin = margin       # 0.3

        self.ranking_loss = nn.MarginRankingLoss(margin=margin)


    def __call__(self, global_feat, face_feat, labels, labels_face, normalize_feature=False):       # [64, 2048],  [64,]
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)         # [64, 64]
        dist_mat_face = euclidean_dist(face_feat, face_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)    # [64,], [64,]
        dist_ap_face, dist_an_face = hard_example_mining(dist_mat_face, labels_face)
        y = dist_an.new().resize_as_(dist_an).fill_(1)              # [64,]
        y_face = dist_an_face.new().resize_as_(dist_an_face).fill_(1)

        loss = self.ranking_loss(dist_an, dist_ap, y) + 0.5 * self.ranking_loss(dist_an_face, dist_ap_face, y_face)
        # loss = self.ranking_loss(dist_an, dist_ap, y) + 0.5 * self.ranking_loss(dist_an_face, dist_ap_face, y_face)
        # loss = self.ranking_loss(dist_an + dist_an_face, dist_ap + dist_ap_face, y + y_face)

        return loss, dist_ap, dist_an

def make_loss(cfg, num_classes):

    tripletwf = TripletWithFaceLoss(0.3)  # triplet loss
    triplet = TripletLoss(0.3)
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    ft_loss = MSELoss()

    def loss_func(score, feat, score_face, feat_face, target, target_face):      # [64, 751],  [64, 512], [64,]
        B = int(score.shape[0] / 2)
        loss_x = xent(score, target)         # 6.618
        loss_x_face = xent(score_face, target_face)
        # loss_t = triplet(feat, target)[0]
        loss_t = tripletwf(feat, feat_face, target, target_face)[0]    # 3.2403
        loss_f = ft_loss(feat[0: B], feat[B:])      # 0.99
        # loss = loss_x + loss_t + loss_f + loss_x_face     # 11.1710

        return loss_x, loss_t, loss_f, loss_x_face

    def loss_func0(score, feat, target):      # [64, 751],  [64, 512], [64,]
        # B = int(score.shape[0] / 2)
        loss_x = xent(score, target)         # 6.618
        loss_t = triplet(feat, target)[0]
        # loss_f = ft_loss(feat[0: B], feat[B:])      # 0.99
        loss = loss_x + loss_t # + loss_f    # 11.1710

        return loss

    return loss_func0

'''
def make_loss(cfg, num_classes):

    triplet = TripletLoss(0.3)  # triplet loss
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    ft_loss = MSELoss()

    def loss_func(score, feat, target):      # [64, 751],  [64, 512], [64,]
        B = int(score.shape[0 ] / 2)
        loss_x = xent(score, target)         # 6.618
        loss_t = triplet(feat, target)[0]    # 3.2403
        loss_f = ft_loss(feat[0: B], feat[B:])      # 0.99
        loss = loss_x + loss_t + loss_f      # 11.1710

        return loss

    return loss_func
'''

def make_loss_face(cfg, num_classes):

    triplet = TripletLoss(0.3)  # triplet loss
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    def loss_func(score, feat, target):      # [64, 751],  [64, 512], [64,]
        loss_x = xent(score, target)         # 6.618
        loss_t = triplet(feat, target)[0]    # 3.2403
        loss = loss_x + loss_t     # 11.1710

        return loss

    return loss_func