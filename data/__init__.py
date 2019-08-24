import os.path as osp
from PIL import Image
from collections import defaultdict
from random import choice
from .datasets import init_multidataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .transforms import build_transforms


def make_dataloaders(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    list_dataset = init_multidataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    num_classes = list_dataset.num_train_pids
    train_set = DFDGANDataset(list_dataset.train, train_transforms)
    train_loader = DataLoader(train_set, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=True, num_workers=num_workers)

    return train_loader, num_classes

class DFDGANDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.pid_index_dict = defaultdict(list)
        self.setid_index_dict = defaultdict(list)
        for index, (_, pid, _, setid) in enumerate(self.dataset):
            self.pid_index_dict[pid].append(index)
            self.setid_index_dict[setid].append(index)
        self.setid_list = list(self.setid_index_dict.keys())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x1_path, x1_pid, x1_camid, x1_setid = self.dataset[index]

        x2_index = self.choose_x2_index(x1_pid, index)
        y1_index, y1_setid = self.choose_y1_index(x1_setid)
        y2_index = self.choose_y2_index(y1_setid, y1_index)

        x2_path, _, _, _ = self.dataset[x2_index]
        y1_path, _, _, _ = self.dataset[y1_index]
        y2_path, _, _, _ = self.dataset[y2_index]

        img_x1 = read_image(x1_path)
        img_x2 = read_image(x2_path)
        img_y1 = read_image(y1_path)
        img_y2 = read_image(y2_path)

        if self.transform is not None:
            img_x1 = self.transform(img_x1)
            img_x2 = self.transform(img_x2)
            img_y1 = self.transform(img_y1)
            img_y2 = self.transform(img_y2)

        return img_x1, img_x2, img_y1, img_y2, x1_pid, y1_setid

    def choose_x2_index(self, x1_pid, index):
        x2_index = choice(self.pid_index_dict[x1_pid])
        while x2_index == index:
            x2_index = choice(self.pid_index_dict[x1_pid])
        return x2_index

    def choose_y1_index(self, x1_setid):
        other_setids = []
        for setid in self.setid_list:
            if setid != x1_setid:
                other_setids.append(setid)
        y1_setid = choice(other_setids)
        y1_index = choice(self.setid_index_dict[y1_setid])
        return y1_index, y1_setid

    def choose_y2_index(self, y1_setid, y1_index):
        y2_index = choice(self.setid_index_dict[y1_setid])
        while y2_index == y1_index:
            y2_index = choice(self.setid_index_dict[y1_setid])

        return y2_index

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

