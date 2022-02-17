import os
from glob import glob
import argparse
import json
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("base_dir")
parser.add_argument("data_dir")

if __name__ == '__main__':
    args = parser.parse_args()

    # BASE_DIR = '/root/class/cmu/LSMA/ls-hw1'
    # DATA_DIR = '/shared/youngkim/dataset'
    BASE_DIR = args.base_dir
    DATA_DIR = args.data_dir
    AUDIO_DIR = os.path.join(DATA_DIR, 'wav_16k')
    wav_file_paths = glob(os.path.join(AUDIO_DIR, '*'))

    train_df = pd.read_csv(os.path.join(DATA_DIR, 'labels/kfold_df.csv'))

    if os.path.exists(f'{DATA_DIR}/datafiles') == False:
        os.mkdir(f'{DATA_DIR}/datafiles')

    # make .json format dataset for 5-fold train/val
    for i in range(5):
        train_wav_list = []
        eval_wav_list = []
        meta = zip(train_df.Id, train_df.Category, train_df.kfold)
        for id, category, fold in meta:
            fold = int(fold)
            wav_path = f'{AUDIO_DIR}/{id}.wav'
            meta_dict = {"wav":wav_path, "labels":category}
            if fold == i:
                eval_wav_list.append(meta_dict)
            else:
                train_wav_list.append(meta_dict)

        print('fold {:d}: {:d} training samples, {:d} test samples'.format(i, len(train_wav_list), len(eval_wav_list)))

        with open(f'{DATA_DIR}/datafiles/med_train_data_'+ str(i) +'.json', 'w') as f:
            json.dump({'data': train_wav_list}, f, indent=1)

        with open(f'{DATA_DIR}/datafiles/med_eval_data_'+ str(i) +'.json', 'w') as f:
            json.dump({'data': eval_wav_list}, f, indent=1)

    # make .json format dataset for whole train data and test
    
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'labels/test_for_students.csv'))

    test_wav_list = []
    for idx, id in enumerate(test_df.Id):
        wav_path = f'{AUDIO_DIR}/{id}.wav'
        meta_dict = {"wav":wav_path, "labels":idx%15}
        test_wav_list.append(meta_dict)

    with open(f'{DATA_DIR}/datafiles/med_test_data.json', 'w') as f:
        json.dump({'data': test_wav_list}, f, indent=1)

    train_wav_list = []
    meta = zip(train_df.Id, train_df.Category)
    for id, category in meta:
        wav_path = f'{AUDIO_DIR}/{id}.wav'
        meta_dict = {"wav":wav_path, "labels":category}
        train_wav_list.append(meta_dict)

    with open(f'{DATA_DIR}/datafiles/med_train_data.json', 'w') as f:
        json.dump({'data': train_wav_list}, f, indent=1)