#!/bin/python

import os
import pickle
import argparse
import sys
import pdb
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix

# Train SVM

parser = argparse.ArgumentParser()
parser.add_argument("feat_dir")
parser.add_argument("feat_dim", type=int)
parser.add_argument("list_videos")
parser.add_argument("output_file")
parser.add_argument("--feat_appendix", default=".csv")

if __name__ == '__main__':
  args = parser.parse_args()

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
  y = np.array(label_list)
  X = np.array(feat_list)

  # pass array for svm training
  # one-versus-rest multiclass strategy
  clf = SVC(cache_size=2000, decision_function_shape='ovr', kernel="rbf", C=50, gamma=1.0)
  result = pd.DataFrame(cross_validate(clf, X, y, cv =5))
  print(result)
  print('mean of cross validation: ', np.mean(result.test_score))

  # save trained SVM in output_file
  pickle.dump(clf, open(args.output_file, 'wb'))
  print('One-versus-rest multi-class SVM trained successfully')
