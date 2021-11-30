import numpy as np
import torch

from decision_transformer.training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask, missions, text_mask = self.get_batch(self.batch_size)
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:,:-1], timesteps, missions, attention_mask=attention_mask, text_attention_mask = text_mask
        )

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            None, action_preds, None,
            None, action_target, None,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.nn.functional.cross_entropy(action_preds, action_target).detach().cpu().item()

        return loss.detach().cpu().item()

    def calculate_val_loss(self, num_steps, batch_fn):
        losses = []
        for _ in range(num_steps):
            # print(_)
            with torch.no_grad():
                # print(0)
                states, actions, rewards, dones, rtg, timesteps, attention_mask, missions, text_mask = batch_fn(self.batch_size)
                # print(1)
                action_target = torch.clone(actions)
                # print(2)
                state_preds, action_preds, reward_preds = self.model.forward(
                    states, actions, rewards, rtg[:,:-1], timesteps, missions, attention_mask=attention_mask, text_attention_mask = text_mask
                )
                # print(3)
                act_dim = action_preds.shape[2]
                action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
                action_target = action_target.reshape(-1)[attention_mask.reshape(-1) > 0]
                loss = self.loss_fn(
                    None, action_preds, None,
                    None, action_target, None,
                )
                # print(loss.item())
                losses.append(loss.detach().cpu().item())
        return np.mean(losses)
