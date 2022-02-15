import os
import argparse
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", defualt='./data')

if __name__ == '__main__':

    args = parser.parse_args()
    train_val_df = pd.read_csv(os.path.join(args.data_dir, 'labels/train_val.csv'))

    gkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, ( _, val_) in enumerate(gkf.split(X=train_val_df, y=train_val_df.Category)):
        train_val_df.loc[val_ , "kfold"] = fold

    train_val_df.to_csv(os.path.join(args.data_dir, 'labels/kfold_df.csv'))