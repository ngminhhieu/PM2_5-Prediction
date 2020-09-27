from pandas import read_csv
import numpy as np
import pandas as pd
import yaml
from lib import constant
import os
from shutil import copyfile

def generate_npz(all_input_features, dataset, output_dir_npz, config_path, seq2seq_path):
    set_config(all_input_features, config_path, output_dir_npz, seq2seq_path)
    dataset = read_csv(dataset, usecols=all_input_features)
    np.savez(output_dir_npz, monitoring_data = dataset)

def set_config(all_input_features, config_path, output_dir_npz, seq2seq_path):
    if os.path.exists(config_path)==False:
        copyfile("config/hanoi/ga_hanoi.yaml", config_path)
    with open(config_path, 'r') as f:
        config = yaml.load(f)    
    config['model']['input_dim'] = len(all_input_features)
    config['base_dir'] = seq2seq_path
    config['data']['dataset'] = output_dir_npz

    with open(config_path, 'w') as f:
        yaml.dump(config, f)