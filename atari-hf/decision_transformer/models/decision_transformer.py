import numpy as np
import torch
import torch.nn as nn

import transformers


from decision_transformer.models.model import TrajectoryModel
# from decision_transformer.models.trajectory_gpt2 import GPT2Model
from transformers import GPT2Model, GPT2Tokenizer,  BertModel, BertTokenizer


class DecisionTransformer(TrajectoryModel):

    """
    This model uses GPT to model (Return_1, state_1, action_1, Return_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=False,
            pre_trained = True,
            freeze_embeddings = False,
            num_frozen_layers = 0,
            bert_encode = False,
            all_states = False,
            keep_mask = False,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
                    n_embd=hidden_size,
                    **kwargs
                )
        self.config = config

        if pre_trained:
            self.transformer = GPT2Model.from_pretrained('gpt2')
            config.n_embd = 768
        else:
            self.config = transformers.GPT2Config(
                        n_embd=hidden_size,
                        **kwargs
                    )
            self.transformer = GPT2Model(self.config)

        self.config.freeze_embeddings = freeze_embeddings
        self.config.bert_encode = bert_encode
        self.config.all_states = all_states
        self.config.keep_mask = keep_mask
        if bert_encode:
            self.bert_encoder = BertModel.from_pretrained('bert-base-uncased').cuda()
        for name, param in self.transformer.named_parameters():
            param_names = ['h.' + str(i) + '.' for i in range(num_frozen_layers)]
            in_name = False
            for param_name in param_names:
                if param_name in name:
                    in_name = True
            if in_name:
                param.requires_grad = False
            # if 'h.1.' in name or 'h.2.' in name or 'h.3.' in name or 'h.0.' in name:
                print(name)

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


        self.hidden_size = config.n_embd

        self.embed_timestep = nn.Embedding(max_ep_len, config.n_embd)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = nn.Sequential(nn.Conv2d(1, 32, 8, stride=4, padding=0), nn.ReLU(),
                                         nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                         nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU(),
                                         nn.Flatten(), nn.Linear(3136, config.n_embd), nn.Tanh())
        self.embed_action = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())

        if self.config.num_words is not None:
            self.embed_text = nn.Embedding(config.num_words, config.n_embd)

        self.embed_ln = nn.LayerNorm(config.n_embd) if config.ln else nn.Identity()

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(config.n_embd, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(config.n_embd, self.act_dim)] + ([nn.Tanh()] if action_tanh else []))
        )
        self.predict_return = torch.nn.Linear(config.n_embd, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps, text_conditioning, attention_mask=None, text_attention_mask = None):

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        text_conditioning = text_conditioning[:, 0]
        if self.config.num_words == None and not self.config.bert_encode:
            text_embeddings = self.transformer.wte.weight[text_conditioning]
        elif not self.config.bert_encode:
            text_embeddings = self.embed_text(text_conditioning)
        elif self.config.all_states:
            text_attention_mask = text_attention_mask[:, 0]
            text_attention_mask = text_attention_mask.cuda()
            text_embeddings = self.bert_encoder(input_ids = text_conditioning, attention_mask = text_attention_mask)['last_hidden_state']
        else:
            text_embeddings = text_conditioning.unsqueeze(1)

        if self.config.freeze_embeddings:
            text_embeddings = text_embeddings.detach()

        # embed each modality with a different head
        states = states.reshape(batch_size, seq_length, 1, 84, 84)
        states = states.reshape(batch_size * seq_length, 1, 84, 84)
        state_embeddings = self.embed_state(states)
        state_embeddings = state_embeddings.reshape(batch_size, seq_length, -1)
        action_embeddings = self.embed_action(actions).squeeze(-2)
        # returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps).squeeze(-2)
        text_context_size = text_embeddings.shape[1]

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings

        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 2*seq_length, self.hidden_size)


        # adds the text-conditioning to the front
        # new view is (T_1, T_2, T_3, ..., R_1, s_1, a_1, etc.)
        stacked_inputs = torch.cat([text_embeddings, stacked_inputs], dim = 1)

        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 2*seq_length)

        # again allow model to see text-conditioning
        if not self.config.keep_mask:
            text_attention_mask = torch.ones((batch_size, text_context_size), dtype = torch.long).to(stacked_inputs.device)
        stacked_attention_mask = torch.cat([text_attention_mask, stacked_attention_mask], dim = 1)



        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        x = x[:, text_context_size:]

        # reshape x so that the second dimension corresponds to
        # predicting returns (0), actions (1), or states (2)
        x = x.reshape(batch_size, seq_length, 2, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        state_preds = self.predict_state(x[:,1])
        action_preds = self.predict_action(x[:,0])
        return_preds = None

        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, text_conditioning, **kwargs):
        # we don't care about the past rewards in this model

        states = states.reshape(1, -1, 1, 84, 84)
        actions = actions.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            # returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # padding
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], 1, 84, 84), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            returns_to_go = None
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)

            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length-actions.shape[1], 1), device=actions.device), actions],
                dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, return_preds = self.forward(states, actions, None, returns_to_go, timesteps, text_conditioning, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]
