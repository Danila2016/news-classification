#!/usr/bin/env python3
""" Late fusion of SVM and RuBert scores """

import pickle
import numpy as np

with open("test_scores_new22.pickle", 'rb') as f:
    bow = pickle.load(f)

with open("test_scores_new26.pickle", 'rb') as f:
    nn = pickle.load(f)


bow = np.array(bow)
nn = np.exp(np.array(nn))

# Check some statistics on scores -> distributions are nearly the same
print(np.mean(bow, 1), np.std(bow, 1))
print(np.mean(bow[bow>0.5]), np.std(bow[bow>0.5]))
print(np.mean(bow[bow<0.5]), np.std(bow[bow<0.5]))
print(np.mean(nn, 1), np.std(nn, 1))
print(np.mean(nn[nn>0.5]), np.std(nn[nn>0.5]))
print(np.mean(nn[nn<0.5]), np.std(nn[nn<0.5]))

scores = 0.3 * bow + 0.7 * nn

with open("test_scores_new26.5.pickle", 'wb') as fw:
    pickle.dump(scores, fw)

final_labels = np.argmax(scores, 0) 

with open("submission26.5.csv", 'w') as fw:
    fw.write("topic,index\n")
    for i, y in enumerate(final_labels):
        fw.write(f"{y},{i}\n")