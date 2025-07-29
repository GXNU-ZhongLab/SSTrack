import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
import json
import re
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings
from lib.train.dataset.depth_utils import get_x_frame


class LasHeR(BaseVideoDataset):
    """ LasHeR dataset(aligned version).

    Publication:
        A Large-scale High-diversity Benchmark for RGBT Tracking
        Chenglong Li, Wanlin Xue, Yaqing Jia, Zhichen Qu, Bin Luo, Jin Tang, and Dengdi Sun
        https://arxiv.org/pdf/2104.13202.pdf

    Download dataset from https://github.com/BUGPLEASEOUT/LasHeR
    """

    def __init__(self, root=None, split='train', dtype='rgbrgb', seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the LasHeR trainingset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().lasher_dir if root is None else root
        assert split in ['train', 'val','all'], 'Only support all, train or val split in LasHeR, got {}'.format(split)
        super().__init__('LasHeR', root)
        self.dtype = dtype

        # all folders inside the root
        self.sequence_list = self._get_sequence_list(split)

        self.data_index = self._create_data_index(split)

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

    def get_name(self):
        return 'lasher'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True # w=h=0 in visible.txt and infrared.txt is occlusion/oov

    def _get_sequence_list(self, split):
        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        file_path = os.path.join(ltr_path, 'data_specs', 'lasher_{}.txt'.format(split))
        with open(file_path, 'r') as f:
            dir_list = f.read().splitlines()
        return dir_list
    
    def _create_data_index(self, split):
        cache_root = os.path.join(os.path.expanduser('~'), '.cache', 'lasher')
        if not os.path.exists(cache_root):
            os.makedirs(cache_root)
        if os.path.exists(os.path.join(cache_root, f'index_{split}.json')):
            with open(os.path.join(cache_root, f'index_{split}.json'), "r") as f:
                lasher_index = json.load(f)
        else:
            lasher_index = {}
            print("saving index for lasher...")
            for seq in self.sequence_list:
                visible_list = os.listdir(os.path.join(self.root, seq, 'visible'))
                visible_list = self._sort(visible_list)
                infrared_list = os.listdir(os.path.join(self.root, seq, 'infrared'))
                infrared_list = self._sort(infrared_list)
                seq_index = {'visible': visible_list, 'infrared': infrared_list}
                lasher_index[seq] = seq_index
            
            with open(os.path.join(cache_root, f'index_{split}.json'), "w") as f:
                json.dump(lasher_index, f)

        return lasher_index

    def _sort(self, x, s='.'):
        x.sort(key=lambda x: int(re.sub('[a-zA-Z]', '', x.split(s)[0])))
        return x

    def _read_bb_anno(self, seq_path):
        # in lasher dataset, visible.txt is same as infrared.txt
        rgb_bb_anno_file = os.path.join(seq_path, "visible.txt")
        # ir_bb_anno_file = os.path.join(seq_path, "infrared.txt")
        rgb_gt = pandas.read_csv(rgb_bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        # ir_gt = pandas.read_csv(ir_bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(rgb_gt)

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def _get_sequence_name(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        return seq_name

    def get_sequence_info(self, seq_id):
        """2022/8/10 ir and rgb have synchronous w=h=0 frame_index"""
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, seq_name, frame_id):
        # Note original filename is chaotic, we rename them
        # rgb_frame_path = os.path.join(seq_path, 'visible', '{:06d}.jpg'.format(frame_id))  # frames start from 0
        # ir_frame_path = os.path.join(seq_path, 'infrared', '{:06d}.jpg'.format(frame_id))

        rgb_video = self.data_index[seq_name]['visible']
        ir_video = self.data_index[seq_name]['infrared']
        try:
            rgb_name = rgb_video[frame_id]
        except Exception as e:
            print('ERROR: Could not find image "{}"'.format(os.path.join(seq_path, 'visible', rgb_name)))
            print(e)
            return None
        try:
            ir_name = ir_video[frame_id]
        except Exception as e:
            print('ERROR: Could not find image "{}"'.format(os.path.join(seq_path, 'infrared', ir_name)))
            print(e)
            return None

        return (os.path.join(seq_path, 'visible', rgb_name), 
                os.path.join(seq_path, 'infrared', ir_name))  # jpg jpg

    def _get_frame(self, seq_path, seq_name, frame_id):
        rgb_frame_path, ir_frame_path = self._get_frame_path(seq_path, seq_name, frame_id)
        img = get_x_frame(rgb_frame_path, ir_frame_path, dtype=self.dtype)
        return img  # cat(visible, infrared): (H, W, 6)

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        seq_name = self._get_sequence_name(seq_id)

        frame_list = [self._get_frame(seq_path, seq_name, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
