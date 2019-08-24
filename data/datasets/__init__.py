from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .bases import BaseImageDataset

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
}

def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)


def init_multidataset(names, *args, **kwargs):
    datasets = []
    base_train_pid = 0
    base_train_camid = 0
    base_query_pid = 0
    base_query_camid = 0
    base_gallery_pid = 0
    base_gallery_camid = 0
    setid = 0
    for name in names:
        datasets.append(init_dataset(name, setid=setid,
                                     base_train_pid=base_train_pid, base_train_camid=base_train_camid,
                                     base_query_pid=base_query_pid, base_query_camid=base_query_camid,
                                     base_gallery_pid=base_gallery_pid, base_gallery_camid=base_gallery_camid,
                                     *args, **kwargs))
        base_train_pid += datasets[-1].num_train_pids
        base_train_camid += datasets[-1].num_train_cams
        base_query_pid += datasets[-1].num_query_pids
        base_query_camid += datasets[-1].num_query_cams
        base_gallery_pid += datasets[-1].num_gallery_pids
        base_gallery_camid += datasets[-1].num_gallery_cams
        setid += 1

    return Multidataset(datasets)

class Multidataset(BaseImageDataset):
    def __init__(self, datasets):
        self.train = []
        self.query = []
        self.gallery = []
        self.num_set = 0
        for dataset in datasets:
            self.train += dataset.train
            self.query += dataset.query
            self.gallery += dataset.gallery
            self.num_set += 1

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
