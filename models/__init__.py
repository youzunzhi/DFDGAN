from .modules import Encoder, DFDGenerator, Discriminator

def make_models(cfg, num_classes):
    E = Encoder(num_classes, cfg.E.LAST_STRIDE, cfg.E.PRETRAIN_PATH, cfg.E.NECK, cfg.TEST.NECK_FEAT, cfg.E.NAME, cfg.E.PRETRAIN_CHOICE)
    Ed = Encoder(num_classes, cfg.ED.LAST_STRIDE, cfg.ED.PRETRAIN_PATH, cfg.ED.NECK, cfg.TEST.NECK_FEAT, cfg.ED.NAME, cfg.ED.PRETRAIN_CHOICE)
    G = DFDGenerator(cfg.G.PRETRAIN_PATH, cfg.G.PRETRAIN_CHOICE, noise_size=cfg.TRAIN.NOISE_SIZE)
    Di = Discriminator(cfg.DI.LAST_STRIDE, cfg.DI.PRETRAIN_PATH, cfg.DI.NECK, cfg.DI.NAME, cfg.DI.PRETRAIN_CHOICE, cfg.DI.DIST_FUNC)
    Dd = Discriminator(cfg.DD.LAST_STRIDE, cfg.DD.PRETRAIN_PATH, cfg.DD.NECK, cfg.DD.NAME, cfg.DD.PRETRAIN_CHOICE, cfg.DD.DIST_FUNC)
    return E, Ed, G, Di, Dd