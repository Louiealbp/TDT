import numpy as np
import torch
import torch.nn as nn

from decision_transformer.models.model import TrajectoryModel
from transformers import GPT2Model, GPT2Tokenizer,  BertModel, BertTokenizer

class MLPBCModel(TrajectoryModel):

    """
    Simple MLP that predicts next action a from past states s.
    """

    def __init__(self, state_dim, act_dim, hidden_size, n_layer, dropout=0.1, max_length=1, **kwargs):
        super().__init__(state_dim, act_dim)

        self.hidden_size = hidden_size
        self.max_length = max_length
        self.bert_encoder = BertModel.from_pretrained('bert-base-uncased').cuda()
        self.image_encoder = nn.Sequential(nn.Conv2d(1, 32, 8, stride=4, padding=0), nn.ReLU(),
                                         nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                         nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
                                         nn.Flatten(), nn.Linear(3136, hidden_size), nn.Tanh())
        self.text_encoder = nn.Sequential(nn.Linear(768, hidden_size), nn.Tanh())

        layers = [nn.Linear((max_length + 1) * hidden_size, hidden_size)]
        for _ in range(n_layer-1):
            layers.extend([
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size)
            ])
        layers.extend([
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.act_dim)
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, text_conditioning, attention_mask=None, text_attention_mask = None):

        states = states[:,-self.max_length:].reshape(states.shape[0], 1, 84, 84)
        states = self.image_encoder(states)

        text_conditioning = text_conditioning[:, 0]
        # text_embeddings = self.bert_encoder(input_ids = text_conditioning.long(), attention_mask = text_attention_mask)['last_hidden_state'][:, 0].detach()
        text_embeddings = self.text_encoder(text_conditioning)

        states = torch.cat([states, text_embeddings], axis = 1)

        actions = self.model(states).reshape(states.shape[0], 1, self.act_dim)

        return None, actions, None

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, text_conditioning, **kwargs):
        # we don't care about the past rewards in this model


        states = states.reshape(1, -1, 1, 84, 84)
        actions = actions.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        states = states[:,-self.max_length:]
        actions = actions[:,-self.max_length:]
        states = torch.cat(
            [torch.zeros((states.shape[0], self.max_length-states.shape[1], 1, 84, 84), device=states.device), states],
            dim=1).to(dtype=torch.float32)

        _, action_preds, return_preds = self.forward(states, actions, None, returns_to_go, timesteps, text_conditioning, attention_mask=None, **kwargs)

        return action_preds[0,-1]
