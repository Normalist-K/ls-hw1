import os
from glob import glob
import argparse
import json
import torchaudio
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("base_dir")
parser.add_argument("data_dir")

if __name__ == '__main__':
    args = parser.parse_args()
    BASE_DIR = args.base_dir
    DATA_DIR = args.data_dir
    EXP_DIR = os.path.join(DATA_DIR, 'exp/best')
    col_name = [i for i in range(15)]
    cums = glob(os.path.join(EXP_DIR, '*/inference/cum_predictions.csv'))
    
    def get_soft(path):
        df = pd.read_csv(path, names=col_name)
        return df.to_numpy()
        
    for i, cum in enumerate(cums):
        if i == 0:
            pred = get_soft(cum)
        else:
            pred += get_soft(cum)
    pred /= 5
    pred = np.argmax(pred, axis=1)
    sub_df = pd.read_csv(os.path.join(DATA_DIR, 'labels/test_for_students.csv'))
    sub_df['Category'] = pred
    sub_df.to_csv(f'{BASE_DIR}/best.csv', index=False)
