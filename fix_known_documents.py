#!/usr/bin/env python3
"""
Set labels of the test documents that appear in the training set

Requires around 4 Gb memory
"""

import csv
import numpy as np
import pandas
from train_kernel_svm import load_test_voc_and_bows, intersection_kernel, dataset2bows, normalize_it

df = pandas.read_pickle('data/train2.p', compression='gzip')

train_data = list(map(str, df['content']))
train_labels = list(map(int, df['label']))

test_labels = []
with open("submission26.5.csv", newline='') as f:
    f.readline()
    reader = csv.reader(f)
    for row in reader:
        test_labels.append(int(row[0]))

voc, test_bows = load_test_voc_and_bows()
print("Normalizing test BOWs")
normalize_it(test_bows)

print("Computing training BOWs")
_, train_bows = dataset2bows(train_data, voc)
print("Normalizing training BOWs")
normalize_it(train_bows)

print("Computing kernel. Please, wait around 10 min") 
K = intersection_kernel(test_bows, train_bows)
count = 0
for i, row in enumerate(K):
    if np.max(row) > 0.95:
        test_labels[i] = train_labels[np.argmax(row)]
        count += 1
print(count, "/", len(test_labels))

with open("submission26.5_patched.csv", 'w') as fw:
        fw.write("topic,index\n")
        for i, y in enumerate(test_labels):
            fw.write(f"{y},{i}\n")
