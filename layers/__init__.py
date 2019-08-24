import torch
import torch.nn as nn
import torch.nn.functional as F

def make_loss_funcs(cfg):
    def loss_G_func(Di_fake_score, Dd_fake_score, img_x1, id_imgs, cyc_imgs, reid_score, target_pid, domain_score, target_setid):
        loss_G_Di = nn.BCELoss()(Di_fake_score, torch.ones_like(Di_fake_score).fill_(1.0))
        loss_G_Dd = nn.BCELoss()(Dd_fake_score, torch.ones_like(Dd_fake_score).fill_(1.0))
        loss_id = nn.L1Loss()(id_imgs, img_x1)
        loss_cyc = nn.L1Loss()(cyc_imgs, img_x1)
        if cfg.TRAIN.STAGE == 1:
            return loss_G_Di * cfg.LOSS_WEIGHT.G_DI + loss_G_Dd * cfg.LOSS_WEIGHT.G_DD + \
                   loss_id * cfg.LOSS_WEIGHT.ID + loss_cyc * cfg.LOSS_WEIGHT.CYC
        elif cfg.TRAIN.STAGE == 2:
            loss_E = nn.CrossEntropyLoss()(reid_score, target_pid)
            loss_Ed = nn.CrossEntropyLoss()(domain_score, target_setid)
            return loss_G_Di * cfg.LOSS_WEIGHT.G_DI + loss_G_Dd * cfg.LOSS_WEIGHT.G_DD + \
                   loss_id * cfg.LOSS_WEIGHT.ID + loss_cyc * cfg.LOSS_WEIGHT.CYC + \
                   loss_E * cfg.LOSS_WEIGHT.E + loss_Ed * cfg.LOSS_WEIGHT.ED

    def loss_Di_func(Di_real_score, Di_fake_score):
        loss_real_id = nn.BCELoss()(Di_real_score, torch.ones_like(Di_real_score).fill_(1.0))
        loss_fake_id = nn.BCELoss()(Di_fake_score, torch.zeros_like(Di_fake_score).fill_(0.0))
        return 0.5 * (loss_real_id + loss_fake_id) * cfg.LOSS_WEIGHT.DI

    def loss_Dd_func(Dd_real_score, Dd_fake_score):
        loss_real_domain = nn.BCELoss()(Dd_real_score, torch.ones_like(Dd_real_score).fill_(1.0))
        loss_fake_domain = nn.BCELoss()(Dd_fake_score, torch.zeros_like(Dd_fake_score).fill_(0.0))
        return 0.5 * (loss_real_domain + loss_fake_domain) * cfg.LOSS_WEIGHT.DD

    return loss_G_func, loss_Di_func, loss_Dd_func