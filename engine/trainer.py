import logging
import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import RunningAverage

from data import make_dataloaders
from models import make_models
from solver import make_optimizers, WarmupMultiStepLR
from layers import make_loss_funcs

global ITER
ITER = 0

def train(cfg, output_dir):
    checkpoint_period = cfg.TRAIN.CHECKPOINT_PERIOD
    device = cfg.TRAIN.DEVICE
    stage = cfg.TRAIN.STAGE
    start_epoch = cfg.TRAIN.START_EPOCH

    logger = logging.getLogger("DFDGAN.train")
    logger.info("Start training, stage "+str(stage))

    train_loader, num_classes = make_dataloaders(cfg)
    E, Ed, G, Di, Dd = make_models(cfg, num_classes)
    optimizer_E, optimizer_Ed, optimizer_G, optimizer_Di, optimizer_Dd = make_optimizers(cfg, E, Ed, G, Di, Dd)
    loss_G_func, loss_Di_func, loss_Dd_func = make_loss_funcs(cfg)
    trainer = make_trainer(cfg, E, Ed, G, Di, Dd,
                         optimizer_E, optimizer_Ed, optimizer_G, optimizer_Di, optimizer_Dd,
                         loss_G_func, loss_Di_func, loss_Dd_func, device=device)

    checkpointer_E, checkpointer_Ed, checkpointer_G, checkpointer_Di, checkpointer_Dd = make_checkpointers(cfg, output_dir, checkpoint_period)
    if stage == 2:
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer_E, {'model': E.state_dict(), 'optimizer': optimizer_E.state_dict()})
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer_Ed, {'model': Ed.state_dict(), 'optimizer': optimizer_Ed.state_dict()})
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer_G, {'model': G.state_dict(), 'optimizer': optimizer_G.state_dict()})
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer_Di, {'model': Di.state_dict(), 'optimizer': optimizer_Di.state_dict()})
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer_Dd, {'model': Dd.state_dict(), 'optimizer': optimizer_Dd.state_dict()})

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    G_scheduler = WarmupMultiStepLR(optimizer_G, cfg.G.SOLVER.STEPS, cfg.G.SOLVER.GAMMA, cfg.G.SOLVER.WARMUP_FACTOR, cfg.G.SOLVER.WARMUP_ITERS, cfg.G.SOLVER.WARMUP_METHOD)
    Di_scheduler = WarmupMultiStepLR(optimizer_Di, cfg.DI.SOLVER.STEPS, cfg.DI.SOLVER.GAMMA, cfg.DI.SOLVER.WARMUP_FACTOR, cfg.DI.SOLVER.WARMUP_ITERS, cfg.DI.SOLVER.WARMUP_METHOD)
    Dd_scheduler = WarmupMultiStepLR(optimizer_Dd, cfg.DD.SOLVER.STEPS, cfg.DD.SOLVER.GAMMA, cfg.DD.SOLVER.WARMUP_FACTOR, cfg.DD.SOLVER.WARMUP_ITERS, cfg.DD.SOLVER.WARMUP_METHOD)
    if stage == 2:
        E_scheduler = WarmupMultiStepLR(optimizer_E, cfg.E.SOLVER.STEPS, cfg.E.SOLVER.GAMMA, cfg.E.SOLVER.WARMUP_FACTOR, cfg.E.SOLVER.WARMUP_ITERS, cfg.E.SOLVER.WARMUP_METHOD)
        Ed_scheduler = WarmupMultiStepLR(optimizer_Ed, cfg.ED.SOLVER.STEPS, cfg.ED.SOLVER.GAMMA, cfg.ED.SOLVER.WARMUP_FACTOR, cfg.ED.SOLVER.WARMUP_ITERS, cfg.ED.SOLVER.WARMUP_METHOD)

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        G_scheduler.step()
        Di_scheduler.step()
        Dd_scheduler.step()
        if stage == 2:
            E_scheduler.step()
            Ed_scheduler.step()

    log_freq = cfg.TRAIN.LOG_FREQ
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss_G')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_loss_Di')
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'avg_loss_Dd')
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_freq == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss_G: {:.3f}, Loss_Di: {:.3f}, Loss_Dd: {:.3f}"
                        .format(engine.state.epoch, ITER, len(train_loader), engine.state.metrics['avg_loss_G'],
                                engine.state.metrics['avg_loss_Di'], engine.state.metrics['avg_loss_Dd'],
                                ))
        if len(train_loader) == ITER:
            ITER = 0

    trainer.run(train_loader, max_epochs=cfg.TRAIN.MAX_EPOCHS)


def make_trainer(cfg, E, Ed, G, Di, Dd,
                 optimizer_E, optimizer_Ed, optimizer_G, optimizer_Di, optimizer_Dd,
                 loss_G_func, loss_Di_func, loss_Dd_func, device='cuda'):
    if device:
        E.to(device)
        Ed.to(device)
        G.to(device)
        Di.to(device)
        Dd.to(device)
        if torch.cuda.device_count() > 1:
            E = nn.DataParallel(E)
            Ed = nn.DataParallel(Ed)
            G = nn.DataParallel(G)
            Di = nn.DataParallel(Di)
            Dd = nn.DataParallel(Dd)

    stage = cfg.TRAIN.STAGE

    def _update(engine, batch):
        G.train()
        Di.train()
        Dd.train()
        if stage==2:
            E.train()
            Ed.train()
        else:
            E.eval()
            Ed.eval()

        optimizer_G.zero_grad()
        optimizer_Dd.zero_grad()
        optimizer_Di.zero_grad()
        if stage==2:
            optimizer_E.zero_grad()
            optimizer_Ed.zero_grad()

        img_x1, img_x2, img_y1, img_y2, target_pid, target_setid = batch
        img_x1, img_x2, img_y1, img_y2, target_pid, target_setid = img_x1.to(device), img_x2.to(device), img_y1.to(device), img_y2.to(device), target_pid.to(device), target_setid.to(device)
        reid_feat, reid_score = E(img_x1)
        domain_feat, domain_score = Ed(img_y1)
        fake_imgs = G(reid_feat, domain_feat)
        Di_fake_score = Di(fake_imgs, img_x1)
        Dd_fake_score = Dd(fake_imgs, img_y1)
        id_imgs = G(E(img_x1)[0], Ed(img_x1)[0])
        cyc_imgs = G(E(fake_imgs)[0], Ed(img_x1)[0])
        loss_G = loss_G_func(Di_fake_score, Dd_fake_score, img_x1, id_imgs, cyc_imgs, reid_score, target_pid, domain_score, target_setid)

        Di_real_score = Di(img_x1, img_x2)
        Dd_real_score = Dd(img_y1, img_y2)
        loss_Di = loss_Di_func(Di_real_score, Di_fake_score)
        loss_Dd = loss_Dd_func(Dd_real_score, Dd_fake_score)

        loss_G.backward(retain_graph=True)
        loss_Di.backward(retain_graph=True)
        loss_Dd.backward()

        optimizer_G.step()
        optimizer_Dd.step()
        optimizer_Di.step()
        if stage == 2:
            optimizer_E.step()
            optimizer_Ed.step()

        return loss_G.item(), loss_Di.item(), loss_Dd.item()
    return Engine(_update)


def make_checkpointers(cfg, output_dir, checkpoint_period):
    checkpointer_E = ModelCheckpoint(output_dir, cfg.E.NAME+'_E', checkpoint_period, n_saved=10, require_empty=False)
    checkpointer_Ed = ModelCheckpoint(output_dir, cfg.ED.NAME+'_Ed', checkpoint_period, n_saved=10, require_empty=False)
    checkpointer_G = ModelCheckpoint(output_dir, cfg.G.NAME+'_G', checkpoint_period, n_saved=10, require_empty=False)
    checkpointer_Di = ModelCheckpoint(output_dir, cfg.DI.NAME+'_Di', checkpoint_period, n_saved=10, require_empty=False)
    checkpointer_Dd = ModelCheckpoint(output_dir, cfg.DD.NAME+'_D', checkpoint_period, n_saved=10, require_empty=False)
    return checkpointer_E, checkpointer_Ed, checkpointer_G, checkpointer_Di, checkpointer_Dd
