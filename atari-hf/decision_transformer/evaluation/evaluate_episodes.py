import numpy as np
import torch
import cv2
from transformers import GPT2Tokenizer, BertModel, BertTokenizer, DistilBertTokenizer

bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# TODO: Use wrapper for environment rather than current set-up

def evaluate_episode_text(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=5000,
        device='cuda',
        target_text=None,
        mode='normal',
        WORD_TO_IDX = None,
        max_len = 20,
        gpt2_text = False,
        bert_encode = False,
        first_state = False,
        all_states = False,
        clip_encode = False,
        train_clip = False,
        clip_model = None
    ):

    model.eval()
    model.to(device=device)

    infos = []

    state = env.reset()
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    state = cv2.resize(state, (84,84))


    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).unsqueeze(0).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    # encode text

    if not gpt2_text and not bert_encode:
        description = target_text.lower().strip().replace(',', '')
        words = description.split(' ')
        words.insert(0, 'START')
        words.append('END')
        words.extend( (max_len - len(words))*['PAD'])
        tokens = [WORD_TO_IDX[word] for word in words]
        text_tokens = np.array(tokens)
        text_tokens = torch.as_tensor(text_tokens).to(device = device).unsqueeze(0).unsqueeze(0)
        text_mask = text_tokens != 0
    elif all_states:
        description = target_text.lower().strip().replace(',', '')
        pad_index = [0]
        indices = bert_tokenizer(description)['input_ids']
        indices.extend(pad_index * (max_len - len(indices)))
        text_tokens = torch.as_tensor(indices).to(device = device).unsqueeze(0).unsqueeze(0)
        text_mask = text_tokens != 0
    elif bert_encode:
        # bert_model = BertModel.from_pretrained('bert-base-uncased')
        # bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        description = target_text.lower().strip().replace(',', '')
        indices = bert_tokenizer(description)['input_ids']
        indices = torch.tensor(indices, dtype = torch.long).unsqueeze(0)
        if not first_state:
            hidden_state = bert_model(input_ids = indices)['last_hidden_state'].detach().cpu().numpy()[:, -1].squeeze()
        else:
            hidden_state = bert_model(input_ids = indices)['last_hidden_state'].detach().cpu().numpy()[:, 0].squeeze()
        text_tokens = torch.as_tensor(hidden_state).to(device = device).unsqueeze(0).unsqueeze(0)
        text_mask = text_tokens != 0
    else:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        pad_index = tokenizer(tokenizer.pad_token)['input_ids']
        description = target_text.lower().strip().replace(',', '')
        indices = tokenizer(description, padding = True)['input_ids']
        indices.extend(pad_index * (max_len - len(indices)))
        text_tokens = torch.as_tensor(indices).to(device = device).unsqueeze(0).unsqueeze(0)


    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)

        # change tokens to float
        action = model.get_action(
            states.to(dtype=torch.float32),
            actions.to(dtype=torch.float32),
            None,
            None,
            timesteps.to(dtype=torch.long),
            text_tokens,
            text_attention_mask = text_mask
        )
        action = top_k_logits(action, 2)

        action = torch.multinomial(torch.nn.functional.softmax(action), 1)
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, info = env.step(action)
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = cv2.resize(state, (84,84))
        infos.append(info)

        cur_state = torch.from_numpy(state).to(device=device).unsqueeze(0).float()
        states = torch.cat([states, cur_state], dim=0)

        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    success = calculate_success(infos, target_text)

    return success, episode_length

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[-1]] = -float('Inf')
    return out

def calculate_success(infos, mission):
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
        success = int(level >= 3)
    elif mission == 'alternate jumping between the second and third ice floes':
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
    elif mission == 'dont stay above the first ice floe':
        successes = 0
        for info in infos:
            if info['labels']['player_y'] > 53:
                successes += 1
        success = float(successes) / float(len(infos))
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
        success = min(1, success / 3000)
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
    elif mission == 'get as far as you can on your first life and then dont move':
        first_life = True
        dont_moves = 0
        for i, info in enumerate(infos):
            if info['ale.lives'] < 4 and first_life:
                first_life = False
                last_lives_idx = i
                score_1 = info['labels']['score_1']
                score_2 = info['labels']['score_2']
                score = (score_1 // 16) * 1000 + (score_1 % 16) * 100 + score_2 // 16 * 10
            if not first_life:
                if info['labels']['player_x'] == 64 and info['labels']['player_y'] == 27:
                     dont_moves += 1
        success = min(1, score / 3000) / 2 + dont_moves / (len(infos) - last_lives_idx) / 2
    return success
