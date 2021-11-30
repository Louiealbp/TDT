"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

from mingpt.utils import sample
import atari_py
from collections import deque
import random
import cv2
from PIL import Image
import gym
# import imageio #uncomment to save videos
from atariari.benchmark.wrapper import AtariARIWrapper

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader
    max_len = 20

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, WORD_TO_IDX):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.writer = SummaryWriter(self.config.ckpt_path)
        self.WORD_TO_IDX = WORD_TO_IDX


        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        # torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            for it, (x, y, m, t, r, lm) in pbar:
                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                m = m.to(self.device)
                t = t.to(self.device)
                r = r.to(self.device)
                lm = lm.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    # logits, loss = model(x, y, r)
                    logits, loss = model(x, y, y, m, t, r, lm)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss
            if is_train:
                train_loss = float(np.mean(losses))
                self.writer.add_scalar('Train/loss', train_loss, epoch_num)

        # best_loss = float('inf')

        best_return = -float('inf')

        self.tokens = 0 # counter used for learning rate decay

        for epoch in range(config.max_epochs):

            run_epoch('train', epoch_num=epoch)
            if self.test_dataset is not None:
                test_loss = run_epoch('test')
                self.writer.add_scalar('Eval/val_loss', test_loss, epoch)

            torch.save(self.model, self.config.ckpt_path + 'encoder.pt')

            # -- pass in target returns (specify if certain missions are being used for fine-tuning)
            if (self.config.game == 'Frostbite-v0' or self.config.game == 'Frostbite-v4') and not self.config.only_jump and not self.config.only_max_points and not self.config.only_reach_level and not self.config.only_certain_points and not self.config.no_labelled_missions and not self.config.only_left_right and not self.config.only_die_level and not self.config.only_certain_ice_floe and not self.config.only_dont_move:
                missions = ['dont move', 'reach level 5', 'stay on the left side', 'try to stay on the right side', 'reach level 2', 'reach level 3', 'reach level 4', 'jump between the second and third ice floes', 'get as close to 1000 score as you can', 'get as close to 500 score as you can', 'get as close to 1500 score as you can', 'get as many points as possible']
            elif self.config.only_jump:
                missions = ['jump between the second and third ice floes']
            elif self.config.only_max_points:
                missions = ['get as many points as possible']
            elif self.config.only_reach_level:
                missions = ['reach level 2', 'reach level 3', 'reach level 4', 'reach level 5']
            elif self.config.only_certain_points:
                missions = ['get as close to 1000 score as you can', 'get as close to 500 score as you can', 'get as close to 1500 score as you can']
            elif self.config.no_labelled_missions:
                missions = []
            elif self.config.only_left_right:
                missions = ['stay on the left side', 'try to stay on the right side']
            elif self.config.only_die_level:
                missions = ['die on the first level', 'die on the second level']
            elif self.config.only_certain_ice_floe:
                missions = ['spend as much time as possible on the first ice floe', 'spend as much time as possible on the fourth ice floe']
            elif self.config.only_dont_move:
                missions = ['dont move']
            elif self.config.game == 'Pong-v0':
                missions = ['go up for 10 steps then go down for 5 steps', 'stay as far away as possible from the enemy player', 'go up for 4 steps then go down for 4 steps', 'go up for 3 steps then go down for 3 steps', 'track the enemy player', 'track the ball', 'go up for 5 steps then go down for 10 steps', 'go up for 3 steps then go down for 10 steps']
            if epoch % self.config.skip_eval == 0:
                for mission in missions:
                    success_rate, frames = self.get_returns(mission = mission, unique = self.config.unique)
                    path = self.config.ckpt_path + '/' + mission.replace(' ', '-') + str(epoch) + '.mp4'
                    # imageio.mimsave(path, frames, fps = 30)
                    self.writer.add_scalar('Eval/' + mission.replace(' ', '-'), success_rate, epoch)

    def get_returns(self, rtg = 0, mission = '', unique = False):
        # prepare model and env
        self.model.train(False)
        if self.config.game == 'Frostbite-v0':
            self.config.game = 'Frostbite-v4'
        env = gym.make(self.config.game)
        env = AtariARIWrapper(env)


        original_mission = mission
        # prepare the mission

        description = mission.lower().strip().replace(',', '')
        if not unique:
            words = description.split(' ')
            words.insert(0, 'START')
            words.append('END')
            words.extend( (self.config.word_size - len(words))*['PAD'])
            tokens = [self.WORD_TO_IDX[word] for word in words]
            orig_mission = np.array(tokens)
        else:
            words = ['START', description, 'END']
            tokens = [self.WORD_TO_IDX[word] for word in words]
            orig_mission = np.array(tokens)
        # store information
        successes = []
        T_rewards, T_Qs = [], []

        done = True
        for i in range(10):
            # set up info to check successes
            frames = []
            infos = []
            final_actions = []

            state = env.reset()
            state = preprocess(state)
            frame = state

            if self.config.state_based:
                state = np.array([0, 0, 0, 0])

            frames.append(frame)
            state = torch.as_tensor(state).type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            mission = torch.as_tensor(orig_mission).to(self.device).unsqueeze(0).unsqueeze(0)
            rtgs = [rtg]
            # first state is from env, and first timestep is 0
            sampled_action = sample(self.model.module, state, 1, temperature=1.0, sample=True, actions=None, top_k = self.config.top_k,
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1),
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device),
                text_conditioning = mission)

            j = 0
            all_missions = mission
            all_states = state
            actions = []
            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                action = sampled_action.cpu().numpy()[0,-1]
                final_actions += [action]
                actions += [sampled_action]

                state, reward, done, info = env.step(action)
                state = preprocess(state)

                if j > 4000:
                    done = True

                frame = state

                if self.config.state_based:
                    state = np.array([info['labels']['player_y'], info['labels']['ball_y'], info['labels']['ball_x'], info['labels']['enemy_y']])

                frames.append(frame)
                infos.append(info)
                reward_sum += reward
                j += 1

                if done:
                    success = self.calculate_success(infos, original_mission, actions)
                    successes.append(success)
                    T_rewards.append(reward_sum)
                    break
                state = torch.as_tensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
                mission = torch.as_tensor(orig_mission).to(self.device).unsqueeze(0).unsqueeze(0)
                all_states = torch.cat([all_states, state], dim=1)
                all_missions = torch.cat([all_missions, mission], dim = 1)

                rtgs += [rtgs[-1] - reward]

                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                sampled_action = sample(self.model.module, all_states, 1, temperature=1.0, sample=True, top_k = self.config.top_k,
                    actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0),
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1),
                    timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)),
                    text_conditioning = mission)

        success_rate = sum(successes)/10.
        print(original_mission)
        print(success_rate)

        self.model.train(True)
        return success_rate, frames

    def calculate_success(self, infos, mission, actions):
        if mission == 'try to stay on the right side':
            successes = 0
            for info in infos:
                if info['labels']['player_x'] > 75:
                    successes += 1
            success = float(successes) / float(len(infos))
        elif mission == 'reach level 2':
            level = 1
            currently_zero = False
            for i, info in enumerate(infos):
                if i < 200:
                    pass
                elif not currently_zero and info['labels']['player_y'] == 0:
                    currently_zero = True
                    level += 1
                elif currently_zero and info['labels']['player_y'] != 0:
                    currently_zero = False
            success = int(level >= 2)
        elif mission == 'reach level 3':
            level = 1
            currently_zero = False
            for i, info in enumerate(infos):
                if i < 200:
                    pass
                elif not currently_zero and info['labels']['player_y'] == 0:
                    currently_zero = True
                    level += 1
                elif currently_zero and info['labels']['player_y'] != 0:
                    currently_zero = False
            success = level >= 3
        elif mission == 'stay on the left side':
            successes = 0
            for info in infos:
                if info['labels']['player_x'] <= 75:
                    successes += 1
            success = float(successes) / float(len(infos))
        elif mission == 'reach level 5':
            level = 1
            currently_zero = False
            for i, info in enumerate(infos):
                if i < 200:
                    pass
                elif not currently_zero and info['labels']['player_y'] == 0:
                    currently_zero = True
                    level += 1
                elif currently_zero and info['labels']['player_y'] != 0:
                    currently_zero = False
            success = int(level >= 5)
        elif mission == 'dont move':
            infos = infos[:4000]
            successes = 0
            for info in infos:
                if info['labels']['player_x'] == 64 and info['labels']['player_y'] == 27:
                    successes += 1
            success = float(successes) / 4000.
        elif mission == 'go up for 10 steps then go down for 5 steps':
            actions = np.array(actions)
            up_actions = actions == 2
            down_actions = actions == 3
            success =  (up_actions.sum() / len(actions)) > 0.6 and (down_actions.sum() / len(actions)) > 0.29
        elif mission == 'go up for 5 steps then go down for 10 steps':
            actions = np.array(actions)
            up_actions = actions == 2
            down_actions = actions == 3
            success =  (up_actions.sum() / len(actions)) > 0.29 and (down_actions.sum() / len(actions)) > 0.6
        elif mission == 'go up for 4 steps then go down for 4 steps' or mission == 'go up for 3 steps then go down for 3 steps':
            actions = np.array(actions)
            up_actions = actions == 2
            down_actions = actions == 3
            success =  (up_actions.sum() / len(actions)) > 0.45 and (down_actions.sum() / len(actions)) > 0.45
        elif mission == 'go up for 3 steps then go down for 10 steps':
            actions = np.array(actions)
            up_actions = actions == 2
            down_actions = actions == 3
            success =  (up_actions.sum() / len(actions)) > 0.2 and (down_actions.sum() / len(actions)) > 0.7
        elif mission == 'jump between the second and third ice floes':
            on_ice_floe = 0 # 0 - not on either, 1 on top, 2 on bottom
            success = 0
            for info in infos:
                if (info['labels']['player_y'] == 79 and on_ice_floe != 1):
                    on_ice_floe = 1
                    success += 1
                elif (info['labels']['player_y'] == 105 and on_ice_floe != 2):
                    on_ice_floe = 2
                    success += 1
                else:
                    success += 0
            success = min(1, success / 50)
        elif mission == 'get as close to 1000 score as you can':
            info = infos[-1]
            score_1 = info['labels']['score_1']
            score_2 = info['labels']['score_2']
            score = (score_1 // 16) * 1000 + (score_1 % 16) * 100 + score_2 // 16 * 10
            success = abs(score - 1000)
            success = max(0, 1 - success / 1000)
        elif mission == 'get as close to 500 score as you can':
            info = infos[-1]
            score_1 = info['labels']['score_1']
            score_2 = info['labels']['score_2']
            score = (score_1 // 16) * 1000 + (score_1 % 16) * 100 + score_2 // 16 * 10
            success = abs(score - 500)
            success = max(0, 1 - success / 500)
        elif mission == 'get as close to 1500 score as you can':
            info = infos[-1]
            score_1 = info['labels']['score_1']
            score_2 = info['labels']['score_2']
            score = (score_1 // 16) * 1000 + (score_1 % 16) * 100 + score_2 // 16 * 10
            success = abs(score - 1500)
            success = max(0, 1 - success / 1500)
        elif mission == 'get as many points as possible':
            info = infos[-1]
            score_1 = info['labels']['score_1']
            score_2 = info['labels']['score_2']
            score = (score_1 // 16) * 1000 + (score_1 % 16) * 100 + score_2 // 16 * 10
            success = score
            success = min(1, score / 3000)
        elif mission == 'die on the first level':
            level = 1
            currently_zero = False
            for i, info in enumerate(infos):
                if i < 200:
                    pass
                elif not currently_zero and info['labels']['player_y'] == 0:
                    currently_zero = True
                    level += 1
                elif currently_zero and info['labels']['player_y'] != 0:
                    currently_zero = False
            success = int(level == 1)
        elif mission == 'die on the second level':
            level = 1
            currently_zero = False
            for i, info in enumerate(infos):
                if i < 200:
                    pass
                elif not currently_zero and info['labels']['player_y'] == 0:
                    currently_zero = True
                    level += 1
                elif currently_zero and info['labels']['player_y'] != 0:
                    currently_zero = False
            success = int(level == 2)
        elif mission == 'spend as much time as possible on the first ice floe':
            success = 0
            for info in infos:
                if info['labels']['player_y'] == 53:
                    success += 1
            success = success / len(infos)
        elif mission == 'spend as much time as possible on the fourth ice floe':
            success = 0
            for info in infos:
                if info['labels']['player_y'] == 131:
                    success += 1
            success = success / len(infos)
        else:
            success = 0

        return success

def preprocess(state):
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = cv2.resize(state, (84,84))
    return state

class Args:
    def __init__(self, game, seed):
        self.device = torch.device('cuda')
        self.seed = seed
        self.max_episode_length = 108e3
        self.game = game
        self.history_length = 4
