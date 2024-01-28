""" Evaluate kernel SVM multiclass classifier based on cross-validation scores """

import pickle
import numpy as np
with open("data/crv.pickle", 'rb') as f:
    data = pickle.load(f)

pred_labels = np.argmax(data['val_scores'], 0)
gt_labels = np.zeros(len(pred_labels), dtype=int)
for i in range(len(data['val_labels'])):
    gt_labels[data['val_labels'][i] == 1] = i
nc = len(data['val_labels'])

print("Multiclass accuracy =", np.sum(pred_labels == gt_labels) / len(gt_labels))

print(data['classes'])
for i in range(nc):
    print(str(i) + " " + " ".join(
        "{:0.2f}".format(np.sum(pred_labels[gt_labels == i] == j) / np.sum(gt_labels == i))
        for j in range(nc)
    ))
