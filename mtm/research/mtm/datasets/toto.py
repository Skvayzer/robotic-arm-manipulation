# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataset used for training a policy. Formed from a collection of
HDF5 files and wrapped into a PyTorch Dataset.
"""

from typing import Any, Callable, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

import pickle
import numpy as np
from PIL import Image
import os

from research.mtm.datasets.base import DatasetProtocol, DataStatistics
from research.mtm.tokenizers.base import TokenizerManager
from .dataset_traj import FrankaDatasetTraj

def get_datasets(
        seq_steps: bool,
        path

):
    print(os.getcwd)
    train_dataset = FrankaDatasetTraj(pickle.load(open(path, 'rb')))
    val_dataset = FrankaDatasetTraj(pickle.load(open(path, 'rb')))
    return train_dataset, val_dataset



class TOTODataset(Dataset, DatasetProtocol):
    def __init__(
            self,
            traj_length: int
    ):
        paths = pickle.load(open('/run/media/cosmos/TOSHIBA EXT/AIRI/data_samples/parsed.pkl', 'rb'))
        self.s = np.concatenate([p['observations'][:-1] for p in paths])
        self.a = np.concatenate([p['actions'][:-1] for p in paths])
        self.sp = np.concatenate([p['observations'][1:] for p in paths])
        self.r = np.concatenate([p['rewards'][:-1] for p in paths])
        self.rollout_score = np.mean([np.sum(p['rewards']) for p in paths])  ### avg of sum of rewards (recorded) of a traj in the expert demos
        self.num_samples = np.sum([p['rewards'].shape[0] for p in paths])
        self._size = len(self.s)
        for path in paths:
            path['images'] = []
            path['depth_images'] = []
            for i in range(path['observations'].shape[0]):
                img = Image.open(os.path.join('data', path['traj_id'], path['cam0c'][i]))  # Assuming RGB for now
                path['images'].append(np.asarray(img))
                img.close()
                depth_img = Image.open(os.path.join('data', path['traj_id'], path['cam0d'][i]))
                path['depth_images'].append(np.asarray(depth_img))
                depth_img.close()
            path['images'] = np.array(path['images']) # (horizon, img_h, img_w, 3)
            path['depth_images'] = np.array(path['depth_images']) # (horizon, img_h, img_w)
        self.paths = paths
        #import pdb; pdb.set_trace()


    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int):
        # img = Image.open(self.paths[idx]['images'])
        return {
            'state': self.s[idx],
            'action': self.a[idx],
            'reward': self.r[idx],
        }



