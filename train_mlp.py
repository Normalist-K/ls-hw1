#!/bin/python

import argparse
import os
import pickle

import numpy as np
from sklearn.neural_network import MLPClassifier

import sys

# Train MLP classifier with labels

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")
parser.add_argument("--hidden_dim", nargs='+', type=int, default=512)

if __name__ == '__main__':

  args = parser.parse_args()
  if isinstance(args.hidden_dim, int):
    hidden_dim = (args.hidden_dim,)
  else:
    hidden_dim = tuple(args.hidden_dim)

  df = pd.read_csv(args.list_videos)

  feat_list, label_list = [], []

  for video_id, label in zip(df.Id, df.Category):
      feat_filepath = os.path.join(args.feat_dir, video_id+args.feat_appendix)
      if os.path.exists(feat_filepath):
          feat_list.append(np.genfromtxt(feat_filepath, delimiter=";", dtype="float"))
          label_list.append(int(label))
      else:
          print(video_id)

  print("number of samples: %s" % len(feat_list))
  X = np.array(feat_list)
  y = np.array(label_list)

  print(f'X: {X.shape}')
  print(f'y: {y.shape}')
  
  clf = MLPClassifier(hidden_layer_sizes=args.hidden_dim,
                      activation="relu",
                      solver="adam",
                      alpha=1e-3)
  clf.fit(X, y)

  # save trained MLP in output_file
  pickle.dump(clf, open(args.output_file, 'wb'))
  print('MLP classifier trained successfully')
