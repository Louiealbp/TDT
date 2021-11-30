import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import sys
from atariari.benchmark.wrapper import AtariARIWrapper
from torch.utils.tensorboard import SummaryWriter
from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_text
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer
from create_dataset import create_dataset, random_shift, create_gpt2_dataset
from torch.utils.data import Dataset
import transformers
from transformers import GPT2Model, GPT2Tokenizer, BertModel, BertTokenizer

transformers.logging.set_verbosity_error()

class StateActionTextDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, missions, timesteps, rtgs, random_shift = False, bert_encode = False, max_text_len = 20, use_indices = False):
        self.block_size = block_size
        self.data = data
        self.actions = actions
        self.vocab_size = int(max(actions) + 1)
        self.done_idxs = done_idxs
        self.missions = missions
        self.timesteps = timesteps
        self.rtgs = rtgs
        try:
            self.num_words = missions.max() + 1
        except:
            self.num_words = 0
        self.bert_encode = bert_encode
        if bert_encode:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.random_shift = random_shift
        self.use_indices = use_indices

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

    # fix attention mask
    def __getitem__(self, idx):
        block_size = self.block_size
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size

        if self.use_indices:
            missions = torch.zeros(block_size, *self.missions.shape[1:], dtype=torch.int64)
            missions[:done_idx-idx] = torch.tensor(self.missions[idx:done_idx], dtype=torch.int64) # (block_size, 20)
        else:
            missions = torch.zeros(block_size, *self.missions.shape[1:], dtype=torch.float32)
            missions[:done_idx-idx] = torch.tensor(self.missions[idx:done_idx], dtype=torch.float32) # (block_size, 20)
        text_attention_mask = missions != 0
        # text_attention_mask = torch.zeros(1)

        states = torch.zeros(block_size, *self.data.shape[1:])
        actions = torch.zeros(block_size, *self.actions.shape[1:], dtype=torch.long)
        timesteps = torch.zeros(block_size, *self.timesteps.shape[1:], dtype=torch.int64)
        rtgs = torch.zeros(block_size, *self.rtgs.shape[1:])
        loss_mask = torch.zeros(block_size, dtype = torch.long)

        states[:done_idx-idx] = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32) # (block_size, 4*84*84)
        actions[:done_idx-idx] = torch.tensor(self.actions[idx:done_idx], dtype=torch.long) # (block_size, 1)
        rtgs[:done_idx-idx] = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32)
        loss_mask[:done_idx-idx] = 1
        if self.random_shift:
            states = random_shift(states)
        states = states.reshape(block_size, -1)
        actions = actions.unsqueeze(1)
        rtgs = rtgs.unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

        return states, actions, missions, timesteps, rtgs, loss_mask, text_attention_mask

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

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    seed = variant['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    env_name = variant['env']
    group_name = f'{exp_prefix}-{env_name}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    path = variant['path']
    writer = SummaryWriter(path)

    if env_name == 'Frostbite-v0':
        env = gym.make('Frostbite-v0')
        env = AtariARIWrapper(env)
        max_ep_len = 10000
        env_targets = ['dont move', 'reach level 5', 'stay on the left side', 'try to stay on the right side', 'reach level 2', 'reach level 3', 'alternate jumping between the second and third ice floes',
        'dont stay above the first ice floe', 'get as many points as possible', 'get as close to 1500 score as you can', 'get as close to 500 score as you can', 'get as close to 1000 score as you can',
        'spend as much time as possible on the first ice floe', 'die on the first level', 'die on the second level', 'get as far as you can on your first life and then dont move', 'spend as much time as possible on the fourth ice floe']
    else:
        raise NotImplementedError()

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n


    print('=' * 50)
    print(f'Starting new experiment: {env_name}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    word_context = variant['word_context']
    game = variant['env']
    random_shift = variant['random_shift']
    gpt2_dataset = variant['gpt2_dataset']

    # make the dataset
    if gpt2_dataset and not variant['unique_missions']:
        obss, actions, returns, done_idxs, rtgs, timesteps, missions, WORD_TO_IDX = create_gpt2_dataset(word_context, game, False, num_missions = variant['num_missions'], bert_encode = variant['bert_encode'], first_state = variant['first_state'], all_states = variant['all_states'], dir = variant['data_dir'])
        val_obss, val_actions, val_returns, val_done_idxs, val_rtgs, val_timesteps, val_missions, val_WORD_TO_IDX = create_gpt2_dataset(word_context, game, False, val = True, bert_encode = variant['bert_encode'], first_state = variant['first_state'], all_states = variant['all_states'], dir = variant['val_dir'])
    else:
        obss, actions, returns, done_idxs, rtgs, timesteps, missions, WORD_TO_IDX = create_dataset(word_context, game, False, unique = variant['unique_missions'], num_missions = variant['num_missions'], dir = variant['data_dir'])
        val_obss, val_actions, val_returns, val_done_idxs, val_rtgs, val_timesteps, val_missions, _ = create_dataset(word_context, game, False, val = True, unique = variant['unique_missions'], dir = variant['val_dir'])


    # create dataloaders
    dataset = StateActionTextDataset(obss, K, actions, done_idxs, missions, timesteps, rtgs, random_shift, bert_encode = variant['bert_encode'], max_text_len = variant['word_context'], use_indices = not variant['first_state'])
    val_dataset = StateActionTextDataset(val_obss, K, val_actions, val_done_idxs, val_missions, val_timesteps, val_rtgs, random_shift, bert_encode = variant['bert_encode'], max_text_len = variant['word_context'], use_indices = not variant['first_state'])
    dataloader = torch.utils.data.DataLoader(dataset, shuffle = True, pin_memory = True, batch_size = batch_size, num_workers = 4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle = True, pin_memory = True, batch_size = batch_size, num_workers = 4)
    data_loader_iter = dataloader.__iter__()
    val_dataloader_iter = val_dataloader.__iter__()
    print("Datasets created")

    # sample from the dataset

    def get_batch(batch_size=256, max_len=K):
        nonlocal data_loader_iter
        try:
            s, a, m, t, rtgs, mask, text_mask = next(data_loader_iter)
        except:
            data_loader_iter = torch.utils.data.DataLoader(dataset, shuffle = True, pin_memory = True, batch_size = batch_size, num_workers = 4).__iter__()
            s, a, m, t, rtgs, mask, text_mask = next(data_loader_iter)
        s = s.to(device)
        a = a.to(device)
        m = m.to(device)
        t = t.to(device)
        rtgs = rtgs.to(device)
        mask = mask.to(device)
        text_mask = text_mask.to(device)
        return s, a, rtgs, None, rtgs, t, mask, m, text_mask

    def get_val_batch(batch_size=256, max_len=K):
        nonlocal val_dataloader_iter
        try:
            s, a, m, t, rtgs, mask, text_mask = next(val_dataloader_iter)
        except:
            val_dataloader_iter = torch.utils.data.DataLoader(val_dataset, shuffle = True, pin_memory = True, batch_size = batch_size, num_workers = 4).__iter__()
            s, a, m, t, rtgs, mask, text_mask = next(val_dataloader_iter)
        s = s.to(device)
        a = a.to(device)
        m = m.to(device)
        t = t.to(device)
        rtgs = rtgs.to(device)
        mask = mask.to(device)
        text_mask = text_mask.to(device)
        return s, a, rtgs, None, rtgs, t, mask, m, text_mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    ret, length = evaluate_episode_text(
                        env,
                        state_dim,
                        1,
                        model,
                        max_ep_len=max_ep_len,
                        device = device,
                        target_text = target_rew,
                        WORD_TO_IDX = WORD_TO_IDX,
                        max_len = word_context,
                        gpt2_text = gpt2_dataset,
                        bert_encode = variant['bert_encode'],
                        first_state = variant['first_state'],
                        all_states = variant['all_states']
                    )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns)
            }
        return fn

    # for creating embedding layers
    if gpt2_dataset:
        num_words = None
    else:
        num_words = len(WORD_TO_IDX)

    if not variant['bc']:
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=dataset.vocab_size,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
            num_words = num_words,
            ln = variant['ln'],
            pre_trained = variant['pre_trained'],
            freeze_embeddings = variant['freeze_embeddings'],
            num_frozen_layers = variant['num_frozen_layers'],
            bert_encode = variant['bert_encode'],
            all_states = variant['all_states'],
            keep_mask = variant['keep_mask']
        )
    else:
        model = MLPBCModel(state_dim=state_dim,
        act_dim=dataset.vocab_size,
        hidden_size = variant['embed_dim'],
        n_layer = variant['n_layer'],
        max_length = 1)

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.nn.functional.cross_entropy(a_hat, a),
        eval_fns=[eval_episodes(tar) for tar in env_targets],
    )


    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )

    max_return = -1e6
    print("Training")
    for iter in range(variant['max_iters']):
        val_loss = trainer.calculate_val_loss(num_steps = 2000, batch_fn = get_val_batch)
        print("Validation loss: ", val_loss)
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)
        else:
            writer.add_scalars('results', outputs, global_step = iter)
            writer.add_scalar('Test/loss', val_loss, global_step = iter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Frostbite-v0')
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=10)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--word_context', type = int, default = 20)
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--random_shift', default = True, action = 'store_false')
    parser.add_argument('--path', default = './logs', type = str)
    parser.add_argument('--gpt2_dataset', default = False, action = 'store_true')
    parser.add_argument('--ln', default = False, action = 'store_true')
    parser.add_argument('--num_missions', default = 0, type = int)
    parser.add_argument('--pre_trained', default = False, action = 'store_true')
    parser.add_argument('--freeze_embeddings', default = False, action = 'store_true')
    parser.add_argument('--num_frozen_layers', default =0, type=int)
    parser.add_argument('--bert_encode', default = False, action = 'store_true')
    parser.add_argument('--first_state', default = False, action = 'store_true')
    parser.add_argument('--all_states', default = False, action = 'store_true')
    parser.add_argument('--keep_mask', default = False, action = 'store_true')
    parser.add_argument('--unique_missions', default = False, action = 'store_true')
    parser.add_argument('--seed', default = 123, type= int)
    parser.add_argument('--bc', default = False, action = 'store_true')
    parser.add_argument('--data_dir', type = str, required = True)
    parser.add_argument('--val_dir', type = str, required = True)
    args = parser.parse_args()

    experiment('gym-experiment', variant=vars(args))
