import argparse
import os, sys, time
from torch.backends import cudnn
from torchvision import transforms
from config import cfg
from utils import setup_logger
from data import make_dataloaders
from models.modules import Encoder, DFDGenerator


def main():
    import pydevd_pycharm
    pydevd_pycharm.settrace('172.26.3.54', port=12345, stdoutToServer=True, stderrToServer=True)
    parser = argparse.ArgumentParser(description="DFDGAN Showing G pic")
    parser.add_argument(
        "--config_file", default="./configs/show_pic.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    datasets_dir = ''
    for dataset_name in cfg.DATASETS.NAMES:
        if datasets_dir != '':
            datasets_dir += '-'
        datasets_dir += dataset_name
    output_dir = os.path.join(output_dir, datasets_dir)
    time_string = 'show_pic[{}]'.format(time.strftime('%Y-%m-%d-%X', time.localtime(time.time())))
    output_dir = os.path.join(output_dir, time_string)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = cfg.TEST.DEVICE
    if device == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.TEST.DEVICE_ID
    cudnn.benchmark = True
    logger = setup_logger("DFDGAN", output_dir, 0)
    logger.info("Running with config:\n{}".format(cfg))

    data_loader, num_classes = make_dataloaders(cfg)
    E = Encoder(num_classes, cfg.E.LAST_STRIDE, cfg.E.PRETRAIN_PATH, cfg.E.NECK, cfg.TEST.NECK_FEAT, cfg.E.NAME, cfg.E.PRETRAIN_CHOICE).to(device)
    Ed = Encoder(num_classes, cfg.ED.LAST_STRIDE, cfg.ED.PRETRAIN_PATH, cfg.ED.NECK, cfg.TEST.NECK_FEAT, cfg.ED.NAME, cfg.ED.PRETRAIN_CHOICE).to(device)
    G = DFDGenerator(cfg.G.PRETRAIN_PATH, cfg.G.PRETRAIN_CHOICE, noise_size=cfg.TRAIN.NOISE_SIZE).to(device)
    for _, batch in enumerate(data_loader):
        img_x1, img_x2, img_y1, img_y2, target_pid, target_setid = batch
        img_x1, img_x2, img_y1, img_y2, target_pid, target_setid = img_x1.to(device), img_x2.to(device), img_y1.to(device), img_y2.to(device), target_pid.to(device), target_setid.to(device)
        g_img = G(E(img_x1)[0], Ed(img_y1)[0])
        img_x1_PIL = transforms.ToPILImage()(img_x1[0].cpu()).convert('RGB')
        img_x1_PIL.save(os.path.join(output_dir, 'img_x1.jpg'))
        img_y1_PIL = transforms.ToPILImage()(img_y1[0].cpu()).convert('RGB')
        img_y1_PIL.save(os.path.join(output_dir, 'img_y1.jpg'))
        g_img_PIL = transforms.ToPILImage()(g_img[0].cpu()).convert('RGB')
        g_img_PIL.save(os.path.join(output_dir, 'g_img.jpg'))
        break
if __name__ == '__main__':
    main()