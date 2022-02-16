import os
import argparse
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", defualt='./data')

if __name__ == '__main__':

    args = parser.parse_args()
    train_val_df = pd.read_csv(os.path.join(args.data_dir, 'labels/train_val.csv'))

    # remove data without audio
    no_audios = ['MzAxOTUwOTU4MTExNjMxNTE4OQ==',
                 'MjU5MzIwNDg2MDQ1MDM1Njg3OQ==',
                 'MjM3NDc0NjAxNTQzOTE3MzE0OA==',
                 'NjE5ODk5NjM1NTUxNjE4NTk2OA==']
    for no_audio in no_audios:
        train_val_df = train_val_df.drop(index=train_val_df[train_val_df.Id == no_audio].index)
    train_val_df = train_val_df.reset_index(drop=True)

    # split data
    gkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, ( _, val_) in enumerate(gkf.split(X=train_val_df, y=train_val_df.Category)):
        train_val_df.loc[val_ , "kfold"] = fold

    # save new csv file
    train_val_df.to_csv(os.path.join(args.data_dir, 'labels/kfold_df.csv'))