# Frostbite

We provide two directories. The first directory (atari) is a light-weight implementation of the Text Decision Transformer which is more explicit, and is based on Chen et. al's Decision Transformer as well as Karpathy's GPT implementation. This first directory contains scripts for replicating the results regarding pre-training. The second directory (atari-hf) allows for adopting hugging face models into the code. This allows for larger models and better performance. This directory contains scripts to replicate the performance vs. baseline plots.

# Installation

These directories contain their own conda environments for installation. This is still a pre-release version of the code, and so is not entirely cleaned up. A future version will contain a wrapped environment for ease of use with the dataset. Note that you may receive errors relating to the atari_py ROMS. Please visit https://github.com/openai/atari-py to see how to fix these issues.

## Downloading datasets

The training dataset (~40G) can be found at: https://drive.google.com/drive/folders/1843RUUsV-WktLwFfD9z2OLn4eNh4rzJk?usp=sharing
and the validation dataset can be found at: https://drive.google.com/drive/folders/1EFXI_LQtLK_aM8x1iLTXdML32b8y_gQS?usp=sharing.

Upon downloading, please add the following lines to your .bashrc (or equivalent):
export DATA_DIR=(your-data-path)
export VAL_DIR=(your-val-dir-path)
This will allow the scripts to run with your local version of the data. Otherwise, feel free to modify the code explicitly with your directories. 
