import csv
import logging
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig
from mingpt.utils import sample
from collections import deque
import random
import torch
import pickle
import blosc
import argparse
from create_dataset import create_dataset, create_unlabelled_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=13)
parser.add_argument('--context_length', type=int, default=20)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--model_type', type=str, default='text_conditioned')
parser.add_argument('--num_demos', type=int, default=1000)
parser.add_argument('--game', type=str, default='Frostbite-v0')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--word_context', type = int, default = 20)
parser.add_argument('--ckpt_path', type = str, default = './logs')
parser.add_argument('--n_layer', type = int, default = 6)
parser.add_argument('--load_dataset', default = False, action = 'store_true')
parser.add_argument('--dataset_path', default = '.', type = str)
parser.add_argument('--save_dataset', default = False, action = 'store_true')
parser.add_argument('--test_word_effectiveness', default = False, action = 'store_true')
parser.add_argument('--top_k', default = 2, type=int)
parser.add_argument('--state_based', default = False, action = 'store_true')
parser.add_argument('--random_shift', default = False, action = 'store_true')
parser.add_argument('--use_offline_data', default = False, action = 'store_true')
parser.add_argument('--use_labelled_offline_data', default = False, action = 'store_true')
parser.add_argument('--only_jump', default = False, action = 'store_true')
parser.add_argument('--zero_shot_jump', default = False, action = 'store_true')
parser.add_argument('--unique_missions', default = False, action = 'store_true')
parser.add_argument('--load_vision_encoder', default = False, action = 'store_true')
parser.add_argument('--vision_encoder_path', default = '.', type = str)
parser.add_argument('--unlabelled_extra_data', default = False, action = 'store_true')
parser.add_argument('--load_embeddings', default = False, action = 'store_true')
parser.add_argument('--only_max_points', default = False, action = 'store_true')
parser.add_argument('--only_certain_points', default = False, action = 'store_true')
parser.add_argument('--only_reach_level', default = False, action = 'store_true')
parser.add_argument('--num_unlabelled_missions', default = 0, type=int)
parser.add_argument('--no_labelled_missions', default = False, action = 'store_true')
parser.add_argument('--only_left_right', default = False, action = 'store_true')
parser.add_argument('--only_dont_move', default = False, action = 'store_true')
parser.add_argument('--only_die_level', default = False, action = 'store_true')
parser.add_argument('--only_certain_ice_floe', default = False, action = 'store_true')
parser.add_argument('--load_model', default = False, action = 'store_true')
parser.add_argument('--bc', default = False, action = 'store_true')
parser.add_argument('--skip_eval', default = 1, type=int)
parser.add_argument('--data_dir', type = str, required = True)
args = parser.parse_args()

set_seed(args.seed)

class StateActionTextDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, missions, timesteps, rtgs, random_shift = False):
        self.block_size = block_size
        self.data = data
        self.actions = actions
        self.vocab_size = max(actions) + 1
        self.done_idxs = done_idxs
        self.missions = missions
        self.timesteps = timesteps
        self.rtgs = rtgs
        self.num_words = missions.max() + 1
        self.random_shift = random_shift


    def __len__(self):
        return len(self.data) - self.block_size

    def dictify(self):
        print(self.done_idxs.shape)
        self.done_idxs = self.done_idxs[:, 0]
        self.done_idxs = set(self.done_idxs.tolist())
        self.done_idxs.add(0)

    def find_before_and_after(self, idx):
        #Find previous done index
        if idx in self.done_idxs:
            prev_done_idx = int(idx)
        else:
            offset = -1
            while True:
                if (idx + offset) in self.done_idxs:
                    prev_done_idx = int(idx + offset)
                    break
                else:
                    offset -= 1
        #Find next done index
        offset = 1
        while True:

            if (idx + offset) in self.done_idxs:
                done_idx = int(idx + offset)
                break
            else:
                offset += 1
        # print(prev_done_idx, done_idx)
        return prev_done_idx, done_idx


    def __getitem__(self, idx):
        block_size = self.block_size
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size

        states = torch.zeros(block_size, *self.data.shape[1:])
        actions = torch.zeros(block_size, *self.actions.shape[1:], dtype=torch.long)
        missions = torch.zeros(block_size, *self.missions.shape[1:], dtype=torch.int64)
        timesteps = torch.zeros(block_size, *self.timesteps.shape[1:], dtype=torch.int64)
        rtgs = torch.zeros(block_size, *self.rtgs.shape[1:])
        loss_mask = torch.zeros(block_size)

        states[:done_idx-idx] = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32) # (block_size, 4*84*84)
        actions[:done_idx-idx] = torch.tensor(self.actions[idx:done_idx], dtype=torch.long) # (block_size, 1)
        missions[:done_idx-idx] = torch.tensor(self.missions[idx:done_idx], dtype=torch.int64) # (block_size, 20)
        rtgs[:done_idx-idx] = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32)
        loss_mask[:done_idx-idx] = 1
        if self.random_shift:
            states = random_shift(states)
        states = states.reshape(block_size, -1)
        actions = actions.unsqueeze(1)
        rtgs = rtgs.unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

        return states, actions, missions, timesteps, rtgs, loss_mask

    def save(self, path):
        dict = {
                'block_size': self.block_size,
                'data': self.data,
                'actions': self.actions,
                'vocab_size': self.vocab_size,
                'done_idxs': self.done_idxs,
                'missions': self.missions,
                'timesteps': self.timesteps,
                'rtgs': self.rtgs
                    }
        np.save(path, dict)

    def load(self, path):
        dict = np.load(path, allow_pickle = True).item()
        self.block_size = dict['block_size']
        self.data = dict['data']
        self.actions = dict['actions']
        self.vocab_size = dict['vocab_size']
        self.done_idxs= dict['done_idxs']
        self.missions = dict['missions']
        self.timesteps = dict['timesteps']
        self.rtgs = dict['rtgs']

def random_shift(states):
    # input is (B, 84, 84)
    new_states = torch.zeros(states.shape[0], 100, 100)
    new_states[:, 8:92, 8:92] = states
    x_offset = np.random.randint(0, 16)
    y_offset = np.random.randint(0, 16)
    new_states = new_states[:, x_offset: x_offset + 84, y_offset: y_offset + 84]
    return new_states

if args.only_jump:
    filter_fn = lambda mission: mission[:12] == 'jump between'
elif args.only_certain_points:
    filter_fn = lambda mission: mission[:15] == 'get as close to'
elif args.only_max_points:
    filter_fn = lambda mission: mission == 'get as many points as possible'
elif args.only_reach_level:
    filter_fn = lambda mission: mission[:11] == 'reach level'
elif args.zero_shot_jump:
    filter_fn = lambda mission: mission != 'jump between the second and third ice floes' and mission != 'jump back and forth between the second and third ice floe'
elif args.no_labelled_missions:
    filter_fn = lambda mission: False
elif args.only_left_right:
    filter_fn = lambda mission: 'left side' in mission or 'right side' in mission
elif args.only_dont_move:
    filter_fn = lambda mission: 'dont move' == mission
elif args.only_die_level:
    filter_fn = lambda mission: 'die on the' in mission
elif args.only_certain_ice_floe:
    filter_fn = lambda mission: 'spend as much time as possible on' in mission
else:
    filter_fn = lambda x: True


if not args.no_labelled_missions:
    obss, actions, returns, done_idxs, rtgs, timesteps, missions, WORD_TO_IDX = create_dataset(args.word_context, args.game, args.state_based, filter_fn = filter_fn, unique = args.unique_missions, dir = args.data_dir)

if args.unlabelled_extra_data:
    if not args.no_labelled_missions:
        unlabelled_obss, unlabelled_actions, unlabelled_returns, unlabelled_done_idxs, unlabelled_rtgs, unlabelled_timesteps, unlabelled_missions, _ = create_unlabelled_dataset(args.word_context, args.game, args.state_based, unique = args.unique_missions, num_missions = args.num_unlabelled_missions, dir = args.data_dir)
        obss = np.concatenate([obss, unlabelled_obss], axis = 0)
        actions = np.concatenate([actions, unlabelled_actions], axis = 0)
        returns = np.concatenate([returns, unlabelled_returns], axis = 0)
        done_idxs = np.concatenate([done_idxs, unlabelled_done_idxs], axis = 0)
        timesteps = np.concatenate([timesteps, unlabelled_timesteps], axis = 0)
        rtgs = np.concatenate([rtgs, unlabelled_rtgs], axis = 0)
        missions = np.concatenate([missions, unlabelled_missions], axis = 0)
    else:
        obss, actions, returns, done_idxs, rtgs, timesteps, missions, WORD_TO_IDX = create_unlabelled_dataset(args.word_context, args.game, args.state_based, unique = args.unique_missions, num_missions = args.num_unlabelled_missions, dir = args.data_dir)


# set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

train_dataset = StateActionTextDataset(obss, args.context_length, actions, done_idxs, missions, timesteps, rtgs, args.random_shift)

if not args.load_model:
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size, args.word_context, len(WORD_TO_IDX.keys()),
                      n_layer=args.n_layer, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=max(train_dataset.timesteps),
                      endpool = False, pixel = False, use_bow = False, test_word_effectiveness = args.test_word_effectiveness, state_based = args.state_based,
                      load_vision_encoder = args.load_vision_encoder, vision_encoder_path = args.vision_encoder_path, load_embeddings = args.load_embeddings)
    model = GPT(mconf)
else:

    model = torch.load(args.vision_encoder_path).module

# initialize a trainer instance and kick off training
epochs = args.epochs
tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
                      num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=max(train_dataset.timesteps), ckpt_path = args.ckpt_path, top_k = args.top_k,
                      word_size = args.word_context, state_based = args.state_based, unique = args.unique_missions, only_jump = args.only_jump, only_max_points = args.only_max_points,
                      only_certain_points = args.only_certain_points, only_reach_level = args.only_reach_level, no_labelled_missions = args.no_labelled_missions,
                      only_left_right = args.only_left_right, only_dont_move = args.only_dont_move, only_die_level = args.only_die_level, only_certain_ice_floe = args.only_certain_ice_floe, skip_eval = args.skip_eval)
trainer = Trainer(model, train_dataset, None, tconf, WORD_TO_IDX)

trainer.train()
