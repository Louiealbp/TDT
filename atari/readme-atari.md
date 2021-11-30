
## Installation

Dependencies can be installed with the following command:

```
conda env create -f conda_env.yml
```

## Downloading datasets

The training dataset (~40G) can be found at: https://drive.google.com/drive/folders/1843RUUsV-WktLwFfD9z2OLn4eNh4rzJk?usp=sharing
and the validation dataset can be found at: https://drive.google.com/drive/folders/1EFXI_LQtLK_aM8x1iLTXdML32b8y_gQS?usp=sharing.

Upon downloading, please add the following lines to your .bashrc (or equivalent):
export DATA_DIR=(your-data-path)
export VAL_DIR=(your-val-dir-path)
This will allow the scripts to run with your local version of the data. Otherwise, feel free to modify the code explicitly with your directories.
## Example usage

Scripts to reproduce the results for experiments can be found in the scripts directory. To run these scripts use the command: bash scripts/(...).sh
