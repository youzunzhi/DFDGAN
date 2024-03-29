from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = "./runs/"

_C.TRAIN = CN()
_C.TRAIN.DEVICE = "cuda"
_C.TRAIN.DEVICE_ID = '0,1'
_C.TRAIN.STAGE = 1
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.MAX_EPOCHS = 120
_C.TRAIN.NOISE_SIZE = 512
_C.TRAIN.CHECKPOINT_PERIOD = 20
_C.TRAIN.LOG_FREQ = 20

_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.0
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
_C.INPUT.PADDING = 10

_C.DATASETS = CN()
_C.DATASETS.NAMES = ['market1501']
_C.DATASETS.ROOT_DIR = ('/home/share/zunzhi')

_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.BATCH_SIZE = 32



_C.E = CN()
_C.E.LAST_STRIDE = 1
_C.E.PRETRAIN_PATH = ''
_C.E.PRETRAIN_CHOICE = 'self'
_C.E.NAME = 'resnet50'
_C.E.NECK = 'bnneck'
_C.E.IF_LABELSMOOTH = 'on'
_C.E.SOLVER = CN()
_C.E.SOLVER.PRETRAIN_CHOICE = 'no'
_C.E.SOLVER.PRETRAIN_PATH = ''
_C.E.SOLVER.OPTIMIZER_NAME = "Adam"
_C.E.SOLVER.BASE_LR = 0.00035
_C.E.SOLVER.BIAS_LR_FACTOR = 1
_C.E.SOLVER.MOMENTUM = 0.9
_C.E.SOLVER.WEIGHT_DECAY = 0.0005
_C.E.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
_C.E.SOLVER.STEPS = (30, 55)
_C.E.SOLVER.GAMMA = 0.1
_C.E.SOLVER.WARMUP_FACTOR = 0.01
_C.E.SOLVER.WARMUP_ITERS = 5
_C.E.SOLVER.WARMUP_METHOD = "linear"


_C.ED = CN()
_C.ED.LAST_STRIDE = 1
_C.ED.PRETRAIN_PATH = ''
_C.ED.PRETRAIN_CHOICE = 'self'
_C.ED.NAME = 'resnet18'
_C.ED.NECK = 'bnneck'
_C.ED.IF_LABELSMOOTH = 'on'
_C.ED.SOLVER = CN()
_C.ED.SOLVER.PRETRAIN_CHOICE = 'no'
_C.ED.SOLVER.PRETRAIN_PATH = ''
_C.ED.SOLVER.OPTIMIZER_NAME = "Adam"
_C.ED.SOLVER.BASE_LR = 0.00035
_C.ED.SOLVER.BIAS_LR_FACTOR = 1
_C.ED.SOLVER.MOMENTUM = 0.9
_C.ED.SOLVER.WEIGHT_DECAY = 0.0005
_C.ED.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
_C.ED.SOLVER.STEPS = (30, 55)
_C.ED.SOLVER.GAMMA = 0.1
_C.ED.SOLVER.WARMUP_FACTOR = 0.01
_C.ED.SOLVER.WARMUP_ITERS = 5
_C.ED.SOLVER.WARMUP_METHOD = "linear"

_C.DI = CN()
_C.DI.LAST_STRIDE = 1
_C.DI.PRETRAIN_PATH = ''
_C.DI.PRETRAIN_CHOICE = 'no'
_C.DI.NAME = 'resnet50'
_C.DI.NECK = 'bnneck'
_C.DI.IF_LABELSMOOTH = 'on'
_C.DI.DIST_FUNC = 'square'
_C.DI.SOLVER = CN()
_C.DI.SOLVER.PRETRAIN_CHOICE = 'no'
_C.DI.SOLVER.PRETRAIN_PATH = ''
_C.DI.SOLVER.OPTIMIZER_NAME = "Adam"
_C.DI.SOLVER.BASE_LR = 0.00035
_C.DI.SOLVER.BIAS_LR_FACTOR = 1
_C.DI.SOLVER.MOMENTUM = 0.9
_C.DI.SOLVER.WEIGHT_DECAY = 0.0005
_C.DI.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
_C.DI.SOLVER.STEPS = (30, 55)
_C.DI.SOLVER.GAMMA = 0.1
_C.DI.SOLVER.WARMUP_FACTOR = 0.01
_C.DI.SOLVER.WARMUP_ITERS = 5
_C.DI.SOLVER.WARMUP_METHOD = "linear"

_C.DD = CN()
_C.DD.LAST_STRIDE = 1
_C.DD.PRETRAIN_PATH = ''
_C.DD.PRETRAIN_CHOICE = 'no'
_C.DD.NAME = 'resnet18'
_C.DD.NECK = 'bnneck'
_C.DD.IF_LABELSMOOTH = 'on'
_C.DD.DIST_FUNC = 'square'
_C.DD.SOLVER = CN()
_C.DD.SOLVER.PRETRAIN_CHOICE = 'no'
_C.DD.SOLVER.PRETRAIN_PATH = ''
_C.DD.SOLVER.OPTIMIZER_NAME = "Adam"
_C.DD.SOLVER.BASE_LR = 0.00035
_C.DD.SOLVER.BIAS_LR_FACTOR = 1
_C.DD.SOLVER.MOMENTUM = 0.9
_C.DD.SOLVER.WEIGHT_DECAY = 0.0005
_C.DD.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
_C.DD.SOLVER.STEPS = (30, 55)
_C.DD.SOLVER.GAMMA = 0.1
_C.DD.SOLVER.WARMUP_FACTOR = 0.01
_C.DD.SOLVER.WARMUP_ITERS = 5
_C.DD.SOLVER.WARMUP_METHOD = "linear"

_C.G = CN()
_C.G.PRETRAIN_PATH = ''
_C.G.PRETRAIN_CHOICE = 'no'
_C.G.NAME = 'DFDGAN'
_C.G.SOLVER = CN()
_C.G.SOLVER.PRETRAIN_CHOICE = 'no'
_C.G.SOLVER.PRETRAIN_PATH = ''
_C.G.SOLVER.OPTIMIZER_NAME = "Adam"
_C.G.SOLVER.BASE_LR = 0.00035
_C.G.SOLVER.BIAS_LR_FACTOR = 1
_C.G.SOLVER.MOMENTUM = 0.9
_C.G.SOLVER.WEIGHT_DECAY = 0.0005
_C.G.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
_C.G.SOLVER.STEPS = (30, 55)
_C.G.SOLVER.GAMMA = 0.1
_C.G.SOLVER.WARMUP_FACTOR = 0.01
_C.G.SOLVER.WARMUP_ITERS = 5
_C.G.SOLVER.WARMUP_METHOD = "linear"

_C.LOSS_WEIGHT = CN()
_C.LOSS_WEIGHT.G_DI = 1.0
_C.LOSS_WEIGHT.G_DD = 1.0
_C.LOSS_WEIGHT.ID = 1.0
_C.LOSS_WEIGHT.CYC = 1.0
_C.LOSS_WEIGHT.E = 1.0
_C.LOSS_WEIGHT.ED = 1.0
_C.LOSS_WEIGHT.DI = 1.0
_C.LOSS_WEIGHT.DD = 1.0

_C.TEST = CN()
_C.TEST.NECK_FEAT = 'after'
_C.TEST.DEVICE = "cuda"
_C.TEST.DEVICE_ID = '0'
