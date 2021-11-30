"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

import numpy as np
from mingpt.networks import ImpalaCNN


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class ResBlock(nn.Module):

    def __init__(self, filters):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(filters, filters, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters, filters, 3, stride=1, padding=1),
        )

    def forward(self, x):
        return x + self.layers(x)

class SimpleResNet(nn.Module):

    def __init__(self, n_input_channels, features_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_input_channels, features_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResBlock(features_dim),
            ResBlock(features_dim),
            ResBlock(features_dim)
        )

    def forward(self, x):
        x = self.net(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h*w).max(dim=-1)[0]
        return x

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, word_size, num_words, **kwargs):
        self.word_size = word_size
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.num_words = num_words
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size * 3 + config.word_size + 1, config.block_size * 3 + config.word_size + 1))
                                     .view(1, 1, config.block_size * 3 + config.word_size + 1, config.block_size * 3 + config.word_size + 1)) ##TODO add support for the number of words to condition on DONE
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.model_type = config.model_type

        # input embedding stem
        if self.config.load_embeddings:
            self.tok_emb = torch.load(self.config.vision_encoder_path).module.tok_emb
        else:
            self.tok_emb = nn.Embedding(config.num_words, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size * 3 + config.word_size + 1, config.n_embd))
        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep + 1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)


        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        if self.config.state_based:
            self.state_encoder = nn.Sequential(nn.Linear(4, 128), nn.ReLU(),
                                               nn.Linear(128, 128), nn.ReLU(),
                                               nn.Linear(128, config.n_embd))
        else:
            if not self.config.load_vision_encoder:
                self.state_encoder = nn.Sequential(nn.Conv2d(1, 32, 8, stride=4, padding=0), nn.ReLU(),
                                         nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                         nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
                                         nn.Flatten(), nn.Linear(3136, config.n_embd), nn.Tanh())
            else:
                self.state_encoder = torch.load(self.config.vision_encoder_path).module.state_encoder
        # not used
        self.ret_emb = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())

        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # state, action, and return
    def forward(self, states, actions, targets=None, text_conditioning = None, timesteps=None, rtgs=None, loss_mask =None):
        # actions: (batch, block_size, 1)
        # targets: (batch, block_size, 1)
        # rtgs: (batch, block_size, 1)
        # timesteps: (batch, block_size, 1)
        # text conditioning: (batch, word_size, -)
        states = states / 255.

        if not self.config.state_based:
            state_embeddings = self.state_encoder(states.reshape(-1, 1, 84, 84))
        else:
            state_embeddings = self.state_encoder(states.reshape(-1, 4))
        state_embeddings = state_embeddings.reshape(states.shape[0], states.shape[1], self.config.n_embd) # (batch, block_size, n_embd)
        if text_conditioning is not None and self.model_type == 'text_conditioned' and actions is not None:
            text_conditioning = text_conditioning[:, 0] # goal is the same for every sampled transition in the same rollout
            text_conditioning = text_conditioning[:, :self.config.word_size]
            text_embeddings = self.tok_emb(text_conditioning)
            action_embeddings = self.action_embeddings(actions.type(torch.long).squeeze(-1))
            token_embeddings = torch.zeros((states.shape[0], self.config.word_size + states.shape[1]*2 - int(targets is None), self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:, :self.config.word_size, :] = text_embeddings
            token_embeddings[:, self.config.word_size::2, :] = state_embeddings
            token_embeddings[:, (self.config.word_size + 1)::2, :] = action_embeddings[:, -states.shape[1] + int(targets is None):, :]

        elif text_conditioning is not None and self.model_type == 'text_conditioned':
            text_conditioning = text_conditioning[:, 0]
            text_conditioning = text_conditioning[:, :self.config.word_size]
            text_embeddings = self.tok_emb(text_conditioning)
            token_embeddings = torch.zeros((states.shape[0], self.config.word_size + states.shape[1], self.config.n_embd), dtype=torch.float32, device=state_embeddings.device)
            token_embeddings[:, :self.config.word_size, :] = text_embeddings
            token_embeddings[:, self.config.word_size::2, :] = state_embeddings
        else:
            raise NotImplementedError()

        batch_size = states.shape[0]
        all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, n_embd

        position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if actions is not None and self.model_type == 'text_conditioned' and text_conditioning is not None:
            logits = logits[:, self.config.word_size::2]
        elif actions is None and self.model_type == 'text_conditioned' and text_conditioning is not None:
            logits = logits[:, self.config.word_size:]
        else:
            raise NotImplementedError()

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            if loss_mask is not None:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction = 'none')
                loss_mask = loss_mask.reshape(-1)
                loss = (loss * loss_mask).mean()
            else:
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss
