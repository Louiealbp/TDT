import numpy as np
import os
import imageio
import torch
from transformers import GPT2Tokenizer, BertModel, BertTokenizer
from PIL import Image

WORD_TO_IDX = {'PAD': 0, 'START': 1, 'END': 2, '1': 3, 'reach': 4, '4000': 5, 'every': 6, 'waste': 7, 'completing': 8, 'close': 9, 'timer': 10, 'no': 11, 'distance': 12, 'sixth': 13, 'minutes': 14, 'regardless': 15,
'there': 16, 'each': 17, 'prioritize': 18, 'entering': 19, 'wait': 20, 'leaving': 21, 'last': 22, 'round': 23, 'stick': 24, 'not': 25, 'birds': 26, 'with': 27, 'to': 28, 'low': 29, 'between': 30, 'play': 31,
'but': 32, 'all': 33, 'number': 34, 'moving': 35, 'more': 36, 'keep': 37, '1500': 38, 'top': 39, 'are': 40, 'furthest': 41, '3000': 42, 'collect': 43, 'time': 44, '6': 45, 'levels': 46, 'seventh': 47,
'of': 48, 'level': 49, 'fifth': 50, 'be': 51, 'least': 52, 'you': 53, 'build': 54, 'six': 55, 'border': 56,
'past': 57, 'as': 58, 'quick': 59, 'going': 60, 'floe': 61, '2000': 62, 'far': 63, 'for': 64, '3': 65,
'soon': 66, '20': 67, 'maximize': 68, 'in': 69, 'sides': 70, 'too': 71, 'much': 72, 'position': 73, 'actions': 74, 'game': 75, 'live': 76, 'downwards': 77, 'jumping': 78, '15000': 79, 'stop': 80, 'first': 81,
 'jump': 82, 'seconds': 83, 'off': 84, 'from': 85, 'above': 86, 'always': 87, 'then': 88, 'floes': 89,
'can': 90, 'four': 91, 'score': 92, 'five': 93, 'complete': 94, 'highest': 95, 'on': 96, 'zero': 97, 'exactly': 98, 'hitting': 99, 'still': 100, 'delay': 101, 'than': 102, 'one': 103, 'starting': 104,
'ice': 105, 'starts': 106, 'built': 107, '10000': 108, 'only': 109, 'life': 110, '1000': 111, 'pass': 112,
'lose': 113, 'without': 114, 'bottom': 115, 'spend': 116, 'out': 117, 'collected': 118, 'into': 119,
'run': 120, 'lives': 121, '5': 122, 'final': 123, 'alternate': 124, 'begins': 125, 'source': 126, 'forth':
 127, '4': 128, 'fourth': 129, '15': 130, 'away': 131, 'by': 132, 'or': 133, 'reward': 134, 'second': 135, 'died': 136, 'farthest': 137, 'row': 138, 'an': 139, 'until': 140, 'hit': 141, 'running': 142,
 'try': 143, 'fewer': 144, 'never': 145, '2': 146, '500': 147, 'your': 148, 'before': 149, 'occupy': 150,
 'fall': 151, 'crabs': 152, 'get': 153, 'at': 154, 'dying': 155, 'have': 156, 'changing': 157,
'igloos': 158, 'fish': 159, 'across': 160, 'colliding': 161, 'two': 162, 'touch': 163, 'finish': 164, 'crab': 165,
 'once': 166, 'bird': 167, 'do': 168, 'right': 169, 'points': 170, 'over': 171, '5000': 172, 'getting': 173,
 'new': 174, 'is': 175, 'when': 176, 'possible': 177, 'rounds': 178, 'the': 179, 'alive': 180, 'times': 181, 'action': 182, 'middle': 183, 'any': 184, 'rows': 185, 'dont': 186, '10': 187, 'playing': 188,
'losing': 189, 'three': 190, 'stand': 191, 'and': 192, 'via': 193, 'after': 194, '2500': 195, 'make': 196,
'7': 197, 'it': 198, 'die': 199, 'back': 200, 'spent': 201, 'quickly': 202, 'jumps': 203, 'left': 204, 'long': 205, 'frostbite': 206, 'go': 207, 'sure': 208, 'survive': 209, 'stay': 210, 'regard': 211, 'a': 212, 'below': 213, 'down': 214, 'change': 215, 'side': 216, 'end': 217, 'third': 218,
'many': 219, 'twice': 220, 'enter': 221, 'let': 222, 'move': 223, 'minimize': 224, 'start': 225, 'water': 226, 'fast': 227, 'few': 228, 'perform': 229, 'next': 230, 'igloo': 231}


def find_final_one(np_array):
    reversed_array = np_array[::-1]
    index = len(reversed_array) - np.argmax(reversed_array) - 1
    return index

def random_shift(states):
    new_states = torch.zeros(states.shape[0], 100, 100)
    new_states[:, 8:92, 8:92] = states
    x_offset = np.random.randint(0, 16)
    y_offset = np.random.randint(0, 16)
    new_states = new_states[:, x_offset: x_offset + 84, y_offset: y_offset + 84]
    return new_states

def create_word_to_idx(dir, unique = False):
    files = os.listdir(dir)
    all_words = set()
    for file in files:
        if file[-3:] == 'npy':
            temp_buffer = np.load(dir + file, allow_pickle= True).item()

            # find all the words that are used
            description = temp_buffer['description']
            description = description.lower().strip().replace(',', '')
            if not unique:
                words = description.split(' ')
                for word in words:
                    all_words.add(word)
            else:
                all_words.add(description)
    all_words.add('maximize')
    all_words.add('reward')
    WORD_TO_IDX = {}
    WORD_TO_IDX['PAD'] = 0
    WORD_TO_IDX['START'] = 1
    WORD_TO_IDX['END'] = 2
    index = 3
    for word in all_words:
        WORD_TO_IDX[word] = index
        index += 1
    return WORD_TO_IDX

def create_dataset(max_len, game, state_based, val = False, unique = False, num_missions = 0, dir = '.'):

    length = 5861673
    obses = np.zeros((length, 84, 84), dtype = np.uint8)
    actions = np.zeros(length)
    dones = np.zeros(length)
    rewards = np.zeros(length)
    missions = np.zeros((length, max_len))
    start_idx = 0

    files = os.listdir(dir)
    # restrict missions
    if num_missions > 0:
        file_numbers = np.random.choice(np.arange(len(files)), size = num_missions, replace = False)
    else:
        file_numbers = np.arange(len(files))
    counter = 0
    for file in files:
        if counter in file_numbers and file[-3:] == 'npy':
            temp_buffer = np.load(dir + file, allow_pickle= True).item()
            total_trajectories = temp_buffer['dones'].sum()
            if total_trajectories == 0:
                # this is a bug where some files did not get a terminal appended at the end
                temp_buffer['dones'][-1] = 1
            assert temp_buffer['dones'].sum() > 0

            # check to make sure the files dont have extra observations between terminal and end of buffer (they likely will)
            final_terminal = find_final_one(temp_buffer['dones'])
            if len(temp_buffer['dones']) - final_terminal > 1000:
                temp_buffer['dones'][-1] = 1
            final_terminal = find_final_one(temp_buffer['dones'])
            temp_buffer['dones'] = temp_buffer['dones'][:final_terminal + 1]
            temp_buffer['obs_ts'] = temp_buffer['obs_ts'][:final_terminal + 1]
            temp_buffer['actions'] = temp_buffer['actions'][:final_terminal + 1]
            temp_buffer['rews'] = temp_buffer['rews'][:final_terminal + 1]
            assert temp_buffer['dones'][-1] == True

            # use word to index to tokenize the descriptions
            description = temp_buffer['description']
            description = description.lower().strip().replace(',', '')
            if not unique:
                words = description.split(' ')
                words.insert(0, 'START')
                words.append('END')
                words.extend( (max_len - len(words))*['PAD'])
                tokens = [WORD_TO_IDX[word] for word in words]
                tokens = np.array(tokens)
                temp_buffer['missions'] = np.stack([tokens] * len(temp_buffer['dones']), axis = 0)
            else:
                words = ['START', description, 'END']
                tokens = [WORD_TO_IDX[word] for word in words]
                tokens = np.array(tokens)
                temp_buffer['missions'] = np.stack([tokens] * len(temp_buffer['dones']), axis = 0)

            curr_width = temp_buffer['dones'].shape[0]
            obses[start_idx: start_idx + curr_width] = temp_buffer['obs_ts']
            actions[start_idx: start_idx + curr_width] = temp_buffer['actions']
            dones[start_idx: start_idx + curr_width] = temp_buffer['dones']
            rewards[start_idx: start_idx + curr_width] = temp_buffer['rews']
            missions[start_idx: start_idx + curr_width] = temp_buffer['missions']
            start_idx += curr_width
        counter += 1

    obses = obses[:start_idx]
    actions = actions[:start_idx]
    dones = dones[:start_idx]
    rewards = rewards[:start_idx]
    missions = missions[:start_idx]

    print(obses.shape)
    print(actions.shape)
    print(dones.shape)
    print(rewards.shape)
    print(missions.shape)

    done_idxs = np.argwhere(dones)
    rtg = np.zeros_like(rewards)
    start_index = 0
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = rewards[start_index:i+1] # includes i
        for j in range(i-1, start_index-1, -1): # start from i-1
            rtg_j = curr_traj_returns[j-start_index:i+1-start_index]
            rtg[j] = rtg_j.sum() # includes i
        start_index = i+1
    start_index = 0
    timesteps = np.zeros(len(actions)+1, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1
    print('max timestep is %d' % max(timesteps))

    return obses, actions, rewards, done_idxs, rtg, timesteps, missions, WORD_TO_IDX

def create_gpt2_dataset(max_len, game, state_based, val = False, num_missions = 0, bert_encode = False, first_state = False, all_states = False, dir = '.'):
    """
    Creates the datset for gpt models (notably uses a different word-to-index mapping and also supports encoding with pre-trained Huggingface models)
    """
    length = 5861673
    obses = np.zeros((length, 84, 84), dtype = np.uint8)
    actions = np.zeros(length)
    dones = np.zeros(length)
    rewards = np.zeros(length)
    missions = np.zeros((length, 768))
    start_idx = 0

    files = os.listdir(dir)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    pad_index = tokenizer(tokenizer.pad_token)['input_ids']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #optionally encode with bert
    if bert_encode:
        bert_model = BertModel.from_pretrained('bert-base-uncased').cuda()
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # restrict missions
    if num_missions > 0:
        file_numbers = np.random.choice(np.arange(len(files)), size = num_missions, replace = False)
    else:
        file_numbers = np.arange(len(files))
    counter = 0

    for file in files:
        if counter in file_numbers and file[-3:] == 'npy':
            temp_buffer = np.load(dir + file, allow_pickle= True).item()

            total_trajectories = temp_buffer['dones'].sum()
            if total_trajectories == 0:
                # this is a bug where some files did not get a terminal appended at the end
                temp_buffer['dones'][-1] = 1
            assert temp_buffer['dones'].sum() > 0

            # check to make sure the files dont have extra observations between terminal and end of buffer (they likely will)
            final_terminal = find_final_one(temp_buffer['dones'])
            if len(temp_buffer['dones']) - final_terminal > 1000:
                temp_buffer['dones'][-1] = 1
            final_terminal = find_final_one(temp_buffer['dones'])
            temp_buffer['dones'] = temp_buffer['dones'][:final_terminal + 1]
            temp_buffer['obs_ts'] = temp_buffer['obs_ts'][:final_terminal + 1]
            temp_buffer['actions'] = temp_buffer['actions'][:final_terminal + 1]
            temp_buffer['rews'] = temp_buffer['rews'][:final_terminal + 1]
            assert temp_buffer['dones'][-1] == True

            if not bert_encode:
                # use word to index to tokenize the descriptions
                description = temp_buffer['description']
                description = description.lower().strip().replace(',', '')
                indices = tokenizer(description, padding = True)['input_ids']
                indices.extend(pad_index * (max_len - len(indices)))
                temp_buffer['missions'] = np.stack([indices] * len(temp_buffer['dones']), axis = 0)

            elif all_states:
                description = temp_buffer['description']
                description = description.lower().strip().replace(',', '')
                pad_index = [0]
                indices = bert_tokenizer(description)['input_ids']
                indices.extend(pad_index * (max_len - len(indices)))
                indices = np.stack([indices] * len(temp_buffer['dones']), axis = 0)
                print(indices.shape)
                temp_buffer['missions'] = indices

            else:
                description = temp_buffer['description']
                description = description.lower().strip().replace(',', '')
                indices = torch.tensor(bert_tokenizer(description)['input_ids'], dtype = torch.long).unsqueeze(0).cuda()

                if not first_state:
                    hidden_state = bert_model(input_ids = indices)['last_hidden_state'].detach().cpu().numpy()[:, -1].squeeze()
                elif first_state:
                    hidden_state = bert_model(input_ids = indices)['last_hidden_state'].detach().cpu().numpy()[:, 0].squeeze()
                temp_buffer['missions'] = np.stack([hidden_state] * len(temp_buffer['dones']), axis = 0)

            curr_width = temp_buffer['dones'].shape[0]
            obses[start_idx: start_idx + curr_width] = temp_buffer['obs_ts']
            actions[start_idx: start_idx + curr_width] = temp_buffer['actions']
            dones[start_idx: start_idx + curr_width] = temp_buffer['dones']
            rewards[start_idx: start_idx + curr_width] = temp_buffer['rews']
            missions[start_idx: start_idx + curr_width] = temp_buffer['missions']
            start_idx += curr_width
        counter += 1

    obses = obses[:start_idx]
    actions = actions[:start_idx]
    dones = dones[:start_idx]
    rewards = rewards[:start_idx]
    missions = missions[:start_idx]

    print(obses.shape)
    print(actions.shape)
    print(dones.shape)
    print(rewards.shape)
    print(missions.shape)

    done_idxs = np.argwhere(dones)
    rtg = np.zeros_like(rewards)
    start_index = 0
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = rewards[start_index:i+1] # includes i
        for j in range(i-1, start_index-1, -1): # start from i-1
            rtg_j = curr_traj_returns[j-start_index:i+1-start_index]
            rtg[j] = rtg_j.sum() # includes i
        start_index = i+1
    start_index = 0
    timesteps = np.zeros(len(actions)+1, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1
    print('max timestep is %d' % max(timesteps))
    WORD_TO_IDX = None
    return obses, actions, rewards, done_idxs, rtg, timesteps, missions, WORD_TO_IDX
