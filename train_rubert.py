#!/usr/bin/env python3
"""
RuBert for multiclass text classification
Code source: https://habr.com/ru/articles/655517/
"""


import sys
import re
import pickle
import csv
csv.field_size_limit(sys.maxsize)

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

import torch
import transformers
import torch.nn as nn
from transformers import AutoModel, BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

tqdm.pandas()

device = torch.device('cuda')

# Load RuBERT model
bert = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased-sentence')

tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased-sentence')

# Load data
df = pd.read_pickle('./train2.p', compression='gzip')
train_data = list(map(str, df['content']))
train_labels = list(map(int, df['label']))

# Shuffle and split data
p = np.random.permutation(len(train_data))
train_data = [train_data[i] for i in p]
train_labels = [train_labels[i] for i in p]

val_data = train_data[-1000:]
val_labels = train_labels[-1000:]

train_data = train_data[:-1000]
train_labels = train_labels[:-1000]

# Load test data
test_data = []
with open("test_news.csv", encoding='utf-8', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        test_data.append(row[0])
test_data = test_data[1:]

# Display number of words in each document for selection of max_length (in tokens)
# See the link for detail and take into account average token length in symbols.
#seq_len = [len(str(x).split()) for x in train_data]
#pd.Series(seq_len).hist(bins = 50)

max_length = 300

# Tokenize data
print("Tokenizing train")
tokens_train = tokenizer.batch_encode_plus(
    train_data,
    max_length = max_length,
    padding = 'max_length',
    truncation = True
)
print("Tokenizing val")
tokens_val = tokenizer.batch_encode_plus(
    val_data,
    max_length = max_length,
    padding = 'max_length',
    truncation = True
)
print("Tokenizing test")
tokens_test = tokenizer.batch_encode_plus(
    test_data,
    max_length = max_length,
    padding = 'max_length',
    truncation = True
)

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels)

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels)

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])

# Create data loaders
batch_size = 8
print("Tensor dataset train")
train_data = TensorDataset(train_seq, train_mask, train_y)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = batch_size)
print("Tensor dataset val")
val_data =  TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size = batch_size)

# Add classification head to bert
for param in bert.parameters():
    param.requires_grad = False

class BERT_Arch(nn.Module):
    
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768,512)
        self.fc2 = nn.Linear(512,9)
        self.softmax = nn.LogSoftmax(dim = 1)
    
    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask = mask, return_dict = False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Prepare for training
model = BERT_Arch(bert)

model = model.to(device)

from transformers import AdamW

optimizer = AdamW(model.parameters(),
               lr= 1e-3)

# In the original code weights were not utilized, and indeed they give -4 accuracy points
"""
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)

print(class_weights)
#[0.8086199  1.31005794]

weights = torch.tensor(class_weights, dtype = torch.float)

weights = weights.to(device)                               
"""

cross_entropy = nn.CrossEntropyLoss()                           # weights not used
epochs = 20

# Training function
def train():
    model.train()
    total_loss, total_accuracy = 0, 0
    total_preds = []
    
    for step, batch in tqdm(enumerate(train_dataloader), total = len(train_dataloader)):
        batch = [r.to(device) for r in batch]
        sent_id,mask,labels = batch
        #print(labels)
        model.zero_grad()
        preds = model(sent_id, mask)
        loss = cross_entropy(preds, labels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds = preds.detach().cpu().numpy()
        total_preds.append(preds)
        
    avg_loss = total_loss / len(train_dataloader)
    total_preds = np.concatenate(total_preds, axis = 0)
    
    return avg_loss, total_preds

# Evaluation function
def evaluate():
    model.eval()
    total_loss, total_accuracy = 0,0
    total_preds = []

    for step, batch in tqdm(enumerate(val_dataloader), total = len(val_dataloader)):
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch
        
        with torch.no_grad():
            preds = model(sent_id, mask)
            loss = cross_entropy(preds, labels)
            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)

    avg_loss = total_loss / len(val_dataloader)
    total_preds = np.concatenate(total_preds, axis = 0)
    
    return avg_loss, total_preds

# Main loop
best_valid_loss = float('inf')

train_losses = []
valid_losses = []

for epoch in range(epochs):
    print('\n Epoch{:} / {:}'.format(epoch+1, epochs))
    
    train_loss, _ = train()
    valid_loss, _ = evaluate()
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    print(f'\nTraining loss: {train_loss:.3f}')
    print(f'Validation loss: {valid_loss:.3f}')

# Load the best model (on validation set)
path = 'saved_weights.pt'
model.load_state_dict(torch.load(path))

# Score validation set
import gc
gc.collect()
torch.cuda.empty_cache()

list_seq = np.array_split(val_seq, 50)
list_mask = np.array_split(val_mask, 50)

predictions = []
for num, elem in enumerate(list_seq):
    with torch.no_grad():
        preds = model(elem.to(device), list_mask[num].to(device))
        predictions.append(preds.detach().cpu().numpy())

flat_preds = [item for sublist in predictions for item in sublist]
pred_scores = np.array(flat_preds)

# Compute predicted labels
pred_labels = np.argmax(pred_scores, 1)

# Evaluate on validation set
print(classification_report(val_labels, pred_labels))

# Score test set
gc.collect()
torch.cuda.empty_cache()

list_seq = np.array_split(test_seq, 50)
list_mask = np.array_split(test_mask, 50)

predictions = []
for num, elem in tqdm(enumerate(list_seq)):
    with torch.no_grad():
        preds = model(elem.to(device), list_mask[num].to(device))
        predictions.append(preds.detach().cpu().numpy())
        
flat_preds = [item for sublist in predictions for item in sublist]
pred_scores = np.array(flat_preds)
final_labels = np.argmax(pred_scores, 1)

# Save test scores
with open("test_scores_new26.pickle", 'wb') as fw:
    pickle.dump(pred_scores.T, fw)

# Save test labels
with open("submission26.csv", 'w') as fw:
    fw.write("topic,index\n")
    for i, y in enumerate(final_labels):
        fw.write(f"{y},{i}\n")


