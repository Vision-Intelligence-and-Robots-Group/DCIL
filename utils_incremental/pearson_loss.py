import torch
from scipy.stats import pearsonr
from torch import nn
from torch.nn import functional as F

def similar_l1(ref_norm_feat, cur_norm_feat, graph_lambda):
    # ref_norm_feat = F.normalize(ref_norm_feat, dim=1)
    # cur_norm_feat = F.normalize(cur_norm_feat, dim=1)
    ref_rank = torch.mm(ref_norm_feat.detach(), (ref_norm_feat.detach().transpose(1, 0)))
    cur_rank = torch.mm(cur_norm_feat, (cur_norm_feat.transpose(1, 0)))
    l1loss = nn.L1Loss()(cur_rank, ref_rank) * graph_lambda
    return l1loss


def similar_l2(ref_norm_feat, cur_norm_feat, graph_lambda):
    # ref_norm_feat = F.normalize(ref_norm_feat, dim=1)
    # cur_norm_feat = F.normalize(cur_norm_feat, dim=1)
    ref_rank = torch.mm(ref_norm_feat.detach(), (ref_norm_feat.detach().transpose(1, 0)))
    cur_rank = torch.mm(cur_norm_feat, (cur_norm_feat.transpose(1, 0)))
    l2loss = nn.MSELoss()(cur_rank, ref_rank) * graph_lambda
    return l2loss


def pearson_loss(ref_norm_feat, cur_norm_feat):
    # ref_norm_feat = F.normalize(ref_norm_feat, dim=1)
    # cur_norm_feat = F.normalize(cur_norm_feat, dim=1)
    ref_rank = torch.mm(ref_norm_feat.detach(), (ref_norm_feat.detach().transpose(1, 0)))
    cur_rank = torch.mm(cur_norm_feat, (cur_norm_feat.transpose(1, 0)))
    x,y = ref_rank.shape

    mref = torch.mean(ref_rank,1)
    mcur = torch.mean(cur_rank,1)

    refm = ref_rank - mref.repeat(y).reshape(y,x).transpose(1,0)
    curm = cur_rank - mcur.repeat(y).reshape(y,x).transpose(1,0)
    r_num = torch.sum(refm*curm, 1)
    r_den = torch.sqrt(torch.sum(torch.pow(refm,2),1)*torch.sum(torch.pow(curm,2),1))
    r = 1 - (r_num / r_den)
    cor = torch.mean(r)
    return cor


def ehg_pearson_loss(cur_norm_feat, ehg_bmu):
    ehg_bmu = torch.Tensor(ehg_bmu).cuda()
    cur_norm_feat = F.normalize(cur_norm_feat, dim=1)
    ehg_bmu = F.normalize(ehg_bmu, dim=1)

    ref_bmu_rank = torch.mm(ehg_bmu, ehg_bmu.transpose(1, 0))
    cur_rank = torch.mm(cur_norm_feat, (cur_norm_feat.transpose(1, 0)))
    x,y = ref_bmu_rank.shape

    mref = torch.mean(ref_bmu_rank,1)
    mcur = torch.mean(cur_rank,1)

    refm = ref_bmu_rank - mref.repeat(y).reshape(y,x).transpose(1,0)
    curm = cur_rank - mcur.repeat(y).reshape(y,x).transpose(1,0)
    r_num = torch.sum(refm*curm, 1)
    r_den = torch.sqrt(torch.sum(torch.pow(refm,2),1)*torch.sum(torch.pow(curm,2),1))
    r = 1 - (r_num / r_den)
    cor = torch.mean(r)
    return cor


def pearson_num(ref_norm_feat, cur_norm_feat):
    ref_rank = torch.mm(ref_norm_feat, (ref_norm_feat.transpose(1, 0)))
    cur_rank = torch.mm(cur_norm_feat, (cur_norm_feat.transpose(1, 0)))
    x,y = ref_rank.shape

    mref = torch.mean(ref_rank,1)
    mcur = torch.mean(cur_rank,1)

    refm = ref_rank - mref.repeat(y).reshape(y,x).transpose(1,0)
    curm = cur_rank - mcur.repeat(y).reshape(y,x).transpose(1,0)
    r_num = torch.sum(refm*curm, 1)
    r_den = torch.sqrt(torch.sum(torch.pow(refm,2),1)*torch.sum(torch.pow(curm,2),1))
    cor = r_num / r_den
    return cor


def pearson_loss_withmask(ref_norm_feat, cur_norm_feat):
    ref_rank = torch.mm(ref_norm_feat.detach(), (ref_norm_feat.detach().transpose(1, 0)))
    cur_rank = torch.mm(cur_norm_feat, (cur_norm_feat.transpose(1, 0)))

    thresh_hold = 0.0
    # thresh_hold = torch.mean(ref_rank)
    mask = ref_rank.gt(thresh_hold)
    ref_rank_mask = ref_rank[mask]
    cur_rank_mask = cur_rank[mask]

    mref = torch.mean(ref_rank_mask,0)
    mcur = torch.mean(cur_rank_mask,0)

    refm = ref_rank_mask - mref
    curm = cur_rank_mask - mcur
    r_num = torch.sum(refm*curm, 0)
    r_den = torch.sqrt(torch.sum(torch.pow(refm,2),0)*torch.sum(torch.pow(curm,2),0))
    r = 1 - (r_num / r_den)
    cor = r
    return cor
