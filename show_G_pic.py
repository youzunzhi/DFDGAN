import argparse
import os, sys, time
from torch.backends import cudnn
from config import cfg
from data import make_dataloaders


def main():
    import pydevd_pycharm
    pydevd_pycharm.settrace('172.26.3.54', port=12345, stdoutToServer=True, stderrToServer=True)
    parser = argparse.ArgumentParser(description="DFDGAN Showing G pic")
    parser.add_argument(
        "--config_file", default="./configs/configs.yml", help="path to config file", type=str
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
    time_string = '[{}]'.format(time.strftime('%Y-%m-%d-%X', time.localtime(time.time())))
    output_dir = os.path.join(output_dir, time_string)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data_loader, num_classes = make_dataloaders(cfg)
    device = 'cuda'


if __name__ == '__main__':
    main()