import torch
from .lr_scheduler import WarmupMultiStepLR

def make_optimizer(cfg_node, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg_node.SOLVER.BASE_LR
        weight_decay = cfg_node.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg_node.SOLVER.BASE_LR * cfg_node.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg_node.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg_node.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg_node.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg_node.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg_node.SOLVER.OPTIMIZER_NAME)(params)

    if cfg_node.SOLVER.PRETRAIN_CHOICE == 'self':
        optimizer.load_state_dict(torch.load(cfg_node.SOLVER.PRETRAIN_PATH))
    return optimizer

def make_optimizers(cfg, E, Ed, G, Di, Dd):
    optimizer_E = make_optimizer(cfg.E, E)
    optimizer_Ed = make_optimizer(cfg.ED, Ed)
    optimizer_G = make_optimizer(cfg.G, G)
    optimizer_Di = make_optimizer(cfg.DI, Di)
    optimizer_Dd = make_optimizer(cfg.DD, Dd)

    return optimizer_E, optimizer_Ed, optimizer_G, optimizer_Di, optimizer_Dd