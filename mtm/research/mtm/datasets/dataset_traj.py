import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from typing import Any, Callable, Dict
import pickle
from pathlib import Path
from research.mtm.datasets.base import DatasetProtocol, DataStatistics
from research.mtm.tokenizers.base import TokenizerManager

def np_to_tensor(nparr, device):
    return torch.from_numpy(nparr).float()

def shift_window(arr, window, np_array=True):
    nparr = np.array(arr) if np_array else list(arr)
    nparr[:-window] = nparr[window:]
    nparr[-window:] = nparr[-1] if np_array else [nparr[-1]] * window
    return nparr

class FrankaDatasetTraj(Dataset):
    def __init__(self, data,
            logs_folder='./',
            subsample_period=1,
            im_h=480,
            im_w=640,
            obs_dim=7,
            action_dim=7,
            H=50,
            top_k=None,
            device='cpu',
            cameras=None,
            img_transform_fn=None,
            noise=None,
            crop_images=False):
        # from multiprocessing import set_start_method
        # try:
        #     set_start_method('spawn')
        # except RuntimeError:
        #     pass
        self.logs_folder = logs_folder
        self.subsample_period = subsample_period
        self.im_h = im_h
        self.im_w = im_w
        self.obs_dim = obs_dim
        self.action_dim = action_dim # not used
        self.H = H
        self.top_k = top_k
        self.demos = data
        self.device = device
        self.cameras = cameras or []
        self.img_transform_fn = img_transform_fn
        self.noise = noise
        self.crop_images = crop_images
        self.pick_high_reward_trajs()
        self.subsample_demos()
        if len(self.cameras) > 0:
            self.load_imgs()
        self.process_demos()

    def pick_high_reward_trajs(self):
        original_data_size = len(self.demos)
        rewards = [traj["rewards"][-1] for traj in self.demos]
        rewards.sort(reverse=True)
        top_rewards = rewards[:int(original_data_size*0.1)]
        self.r_init = sum(top_rewards) / len(top_rewards)
        if self.top_k == None: # assumed using all successful traj (reward > 0)
            self.demos = [traj for traj in self.demos if traj['rewards'][-1] > 0]
            print(f"Using {len(self.demos)} number of successful trajs. Total trajs: {original_data_size}")
        elif self.top_k == 1: # using all data
            pass
        else:
            self.demos = sorted(self.demos, key=lambda x: x['rewards'][-1], reverse=True)
            top_idx_thres = int(self.top_k * len(self.demos))
            print(f"picking top {self.top_k * 100}% of trajs: {top_idx_thres} from {original_data_size}")
            self.demos = self.demos[:top_idx_thres] 
        random.shuffle(self.demos)

    def subsample_demos(self):
        for traj in self.demos:
            for key in ['cam0c', 'cam0d', 'observations', 'actions', 'terminated', 'rewards', 'embeddings']:
                if key == 'observations':
                    traj[key] = traj[key][:, :self.obs_dim]
                if key == 'rewards':
                    rew = traj[key][-1]
                    traj[key] = traj[key][::self.subsample_period]
                    traj[key][-1] = rew
                else:
                    traj[key] = traj[key][::self.subsample_period]

    def process_demos(self):
        self.idx = []
        for i, traj in enumerate(self.demos):
            start = 0
            end = self.H
            while end < traj['actions'].shape[0]:
                self.idx.append((i, start, end))
                start += self.H
                end += self.H
            self.idx.append((i, start, traj['actions'].shape[0]))


            # if traj['actions'].shape[0] > self.H:
            #     for start in range(traj['actions'].shape[0] - self.H + 1):
            #         self.idx.append((i, start, start + self.H))
            # else:
            #     self.idx.append((traj["traj_id"], 0, traj['actions'].shape[0]))

        # if self.cameras:
        #     images = []
        #     for traj in self.demos:
        #         if traj['actions'].shape[0] > self.H:
        #             for start in range(traj['actions'].shape[0] - self.H + 1):
        #                 images.append(traj['images'][start])
        #         else:
        #             images.append(traj['images'][0])
        #     self.images = images
        # self.labels = self.labels.reshape([self.labels.shape[0], -1]) # flatten actions to (#trajs, H * action_dim)

    def load_imgs(self):
        print("Start loading images...")
        for path in self.demos:
            path['images'] = []
            for i in range(path['observations'].shape[0]):
                img_path = os.path.join(self.logs_folder, os.path.join('data', path['traj_id'], path['cam0c'][i]))
                path['images'].append(img_path)
        print("Finished loading images.")


    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        datapoint = dict()
        traj_id, start, end = self.idx[idx]
        traj = self.demos[traj_id]
        for key in ['observations', 'actions', 'rewards', 'embeddings']:
            if key == "rewards":
                datapoint[key] = np.zeros((self.H))
                reward = np.empty(end-start)
                reward.fill(self.r_init)
                reward[-1] -= traj[key][-1]
                datapoint[key][:end-start] = reward
                datapoint[key] = np.expand_dims(datapoint[key], axis=-1)
            else:
                datapoint[key] = np.zeros((self.H, traj[key].shape[1]))
                datapoint[key][:end-start, :] = traj[key][start:end, :]
            datapoint[key] = np_to_tensor(datapoint[key], self.device)

        # if self.noise:
        #     datapoint['inputs'] += torch.randn_like(datapoint['inputs']) * self.noise
        # if self.cameras:
        #     for _c in self.cameras:
        #         try:
        #             img = Image.open(self.images[idx])
        #         except:
        #             print("\n***Image path does not exist. Set the image directory as logs_folder in the config.")
        #             raise
        #         datapoint[_c] = img.crop((200, 0, 500, 400)) if self.crop_images else img
        #         datapoint[_c] = (self.img_transform_fn(datapoint[_c]) if self.img_transform_fn
        #                          else datapoint[_c])
        #         img.close()
        return datapoint
    

    def eval_logs(
        self, model: Callable, tokenizer_manager: TokenizerManager
    ) -> Dict[str, Any]:
        return {}

    def trajectory_statistics(self) -> Dict[str, DataStatistics]:
        # if self.use_avg:
        #     save_path = f"/tmp/toto/toto_{self.name}_statistics_avg.pkl"
        # else:
        #     save_path = f"/tmp/toto/toto_{self.name}_statistics_{self.discount}.pkl"

        # if self.use_achieved:
        #     save_path = save_path.replace(".pkl", "_achieved.pkl")
        # save_path = Path(save_path)

        # if save_path.exists():
        #     with open(save_path, "rb") as f:
        #         ret_dict = pickle.load(f)
        #         return ret_dict

        keys = ["observations", "actions", "rewards"]
        observations, actions, rewards = np.empty([0, 7]), np.empty([0, 7]), np.empty([0])

        for traj in self.demos:
            observations = np.vstack((observations, traj['observations']))
            actions = np.vstack((actions, traj['actions']))
            rewards = np.hstack((rewards, traj['rewards']))
        stats = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
        }

        data_stats = {}
        for key in keys:
            d = stats[key]
            # B, T, X = d.shape
            # d = d.reshape(B * T, X)
            data_stats[key] = DataStatistics(
                np.mean(d, axis=0),
                np.std(d, axis=0),
                np.min(d, axis=0),
                np.max(d, axis=0),
            )

        # save_path.parent.mkdir(parents=True, exist_ok=True)
        # with open(save_path, "wb") as f:
        #     pickle.dump(data_stats, f)
        return data_stats
        # return {"states": 0, 
        #         "actions": 0,
        #         "rewards": 0,
        #         "returns": 0}