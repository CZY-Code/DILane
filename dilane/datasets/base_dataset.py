from cProfile import label
import os.path as osp
import os
import cv2
from torch.utils.data import Dataset
import logging
from .registry import DATASETS
from .process import Process
from dilane.utils.visualization import imshow_lanes
from mmcv.parallel import DataContainer as DC


@DATASETS.register_module
class BaseDataset(Dataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.data_root = data_root
        self.training = 'train' in split
        self.processes = Process(processes, cfg)

    def view(self, predictions, img_metas, labels):
        img_metas = [item for img_meta in img_metas.data for item in img_meta]
        for lanes, img_meta, label in zip(predictions, img_metas, labels):
            img_name = img_meta['img_name']
            img = cv2.imread(osp.join(self.data_root, img_name))
            out_file = osp.join(self.cfg.work_dir, 'visualization',
                                img_name.replace('/', '_'))
            lanes_gt = [lane.to_array(self.cfg) for lane in label]
            lanes = [lane.to_array(self.cfg) for lane in lanes] #具体实现在lane.py里面
            imshow_lanes(img, lanes, lanes_gt, out_file=out_file)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        img = cv2.imread(data_info['img_path'])
        img = img[self.cfg.cut_height:, :, :] #[H, W, C]
        sample = data_info.copy()
        sample.update({'img': img})

        if self.training:
            label = cv2.imread(sample['mask_path'], cv2.IMREAD_UNCHANGED)
            if len(label.shape) > 2:
                label = label[:, :, 0]
            label = label.squeeze()
            label = label[self.cfg.cut_height:, :]
            sample.update({'mask': label})

            if self.cfg.cut_height != 0:
                new_lanes = []
                for i in sample['lanes']:
                    lanes = []
                    for p in i:
                        lanes.append((p[0], p[1] - self.cfg.cut_height))
                    new_lanes.append(lanes)
                sample.update({'lanes': new_lanes})
        else: #validate czy
            new_lanes = []
            for i in sample['lanes']:
                lanes = []
                for p in i:
                    lanes.append((p[0], p[1] - self.cfg.cut_height))
                new_lanes.append(lanes)
            sample.update({'lanes': new_lanes})
            
        sample = self.processes(sample)
        meta = {'full_img_path': data_info['img_path'],
                'img_name': data_info['img_name']}
        meta = DC(meta, cpu_only=True)
        sample.update({'meta': meta})

        return sample
