#!/usr/bin/env python
# coding: utf-8

# In this notebook we will train a deep learning model using all the data available !
# * preprocessing : I encoded the smiles of all the train & test set and saved it [here](https://www.kaggle.com/datasets/ahmedelfazouan/belka-enc-dataset) , this may take up to 1 hour on TPU.
# * Training & Inference : I used a simple 1dcnn model trained on 20 epochs.
# 
# How to improve :
# * Try a different architecture : I'm able to get an LB score of 0.604 with minor changes on this architecture.
# * Try another model like Transformer, or LSTM.
# * Train for more epochs.
# * Add more features like a one hot encoding of bb2 or bb3.
# * And of course ensembling with GBDT models.


import gc
import os
import pickle
import random
import joblib
import pandas as pd
# import polars as pd
from tqdm import tqdm
from sklearn.metrics import average_precision_score as APS
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch.nn.functional as F
import math

from module import network, dataset, util
from importlib import reload


# In[2]:


class Config:
    PREPROCESS = False
    KAGGLE_NOTEBOOK = False
    DEBUG = True
    
    SEED = 42
    EPOCHS = 9
    BATCH_SIZE = 4096
    LR = 1e-3
    WD = 1e-6
    PATIENCE = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NBR_FOLDS = 15
    SELECTED_FOLDS = [0]
    EARLY_STOPPING = False
    
    
if Config.DEBUG:
    n_rows = 10**6
else:
    n_rows = None

# print Config
print("Config: ", Config.__dict__)
print("n_rows: ", n_rows)


# In[3]:


if Config.KAGGLE_NOTEBOOK:
    RAW_DIR = "/kaggle/input/leash-BELKA/"
    PROCESSED_DIR = "/kaggle/input/belka-enc-dataset"
    OUTPUT_DIR = ""
    MODEL_DIR = ""
else:
    RAW_DIR = "../data/raw/"
    PROCESSED_DIR = "../data/processed/"
    OUTPUT_DIR = "../data/result/"
    MODEL_DIR = "../models/"

TRAIN_DATA_NAME = "train_enc.parquet"



def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seeds(seed=Config.SEED)

train_file_list = [f"../data/chuncked-dataset/local_train_enc_{i}.parquet" for i in range(10)]
train_file_list



# train = pl.read_parquet(os.path.join(PROCESSED_DIR, TRAIN_DATA_NAME), n_rows=n_rows)
test = pl.read_parquet(os.path.join(PROCESSED_DIR, 'test_enc.parquet'), n_rows=n_rows)
# train = train.to_pandas()
test = test.to_pandas()



    
def prepare_data(train, train_idx, valid_idx, features, targets, device):
    """
    データの準備を行う関数
    """
    X_train = torch.tensor(train.loc[train_idx, features].values, dtype=torch.float32).to(device)
    y_train = torch.tensor(train.loc[train_idx, targets].values, dtype=torch.float32).to(device)
    X_val = torch.tensor(train.loc[valid_idx, features].values, dtype=torch.float32).to(device)
    y_val = torch.tensor(train.loc[valid_idx, targets].values, dtype=torch.float32).to(device)
    
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    return train_loader, valid_loader, X_val, y_val


def prepare_dataloader(train, val, features, targets, device):
    X_train = torch.tensor(train.loc[:, features].values, dtype=torch.float32).to(device)
    y_train = torch.tensor(train.loc[:, targets].values, dtype=torch.float32).to(device)
    X_val = torch.tensor(val.loc[:, features].values, dtype=torch.float32).to(device)
    y_val = torch.tensor(val.loc[:, targets].values, dtype=torch.float32).to(device)
    
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    return train_loader, valid_loader, X_val, y_val




class Trainer:
    def __init__(self, model, criterion, optimizer, device, patience):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.patience = patience

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def validate(self, valid_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(valid_loader.dataset)
        return val_loss

    def train(self, train_file_list, epochs):
        best_val_loss = float('inf')
        patience_counter = 0
        # valを固定 
        val = pl.read_parquet(train_file_list[9], n_rows=n_rows).to_pandas()
        # print("loaded val data", val.shape, train_file_list[9])
        for epoch in range(epochs):
            train = pl.read_parquet(train_file_list[epoch % 9], n_rows=n_rows).to_pandas()
            # print("loaded train data", train.shape, train_file_list[epoch % 9])
            
            train_loader, valid_loader, X_val, y_val = prepare_dataloader(train, val, FEATURES, TARGETS, Config.DEVICE)
            
            epoch_loss = self.train_epoch(train_loader)
            val_loss = self.validate(valid_loader)
            # APSも計算
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f},  Val Loss: {val_loss:.4f} ')
            
            if Config.EARLY_STOPPING:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pt'))
                    print(f"model saved")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print('Early stopping')
                        break
            else:
                torch.save(self.model.state_dict(), os.path.join(MODEL_DIR, f'best_model.pt'))
                print(f"model saved")
                
        return best_val_loss

    # 1行ずつ予測(メモリ節約)
    def predict(self, data_loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs.to(self.device)
                outputs = torch.sigmoid(self.model(inputs))  # apply sigmoid
                predictions.append(outputs.cpu().numpy())
        return np.concatenate(predictions)

def predict_in_batches(model, data, batch_size):
    model.eval()  # Set model to evaluation mode
    preds = []
    for i in range(0, data.size(0), batch_size):
        batch = data[i:i+batch_size].to(Config.DEVICE)
        with torch.no_grad():
            batch_preds = torch.sigmoid(model(batch))  # apply sigmoid
        preds.append(batch_preds.detach().cpu())
    return torch.cat(preds, dim=0)



# なぜかスコアが0.027程度．コピーしたノートは0.39くらい．データ数の違い？ロスが下がらない
# balanced dataを使って．cvが0.2程度．kaggleノートでも0.04程度．どこが原因かわからない
# BCEwithLogitsLossでロスはまえより下がるようになったが，スコアはあがらない
#　原因はweight decayが高すぎた．10**-6にした
"TODO:適合不足の可能性があるので，訓練スコアを見る "
"TODO: モニター指標としてAPSを使う"

# 定数やモデルの定義は適宜修正してください
FEATURES = [f'enc{i}' for i in range(142)]
TARGETS = ['bind1', 'bind2', 'bind3']



pos_weight = torch.tensor([215, 241, 136], device=Config.DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# model = network.TransformerModel(device=Config.DEVICE).to(Config.DEVICE)
model = network.ImprovedCNNModel().to(Config.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WD)

# StratifiedKFoldの設定
skf = StratifiedKFold(n_splits=Config.NBR_FOLDS, shuffle=True, random_state=42)
all_preds = []


# データの準備
train = pl.read_parquet(train_file_list[0], n_rows=n_rows).to_pandas()
trainer = Trainer(model, criterion, optimizer, Config.DEVICE, Config.PATIENCE)
trainer.train(train_file_list, Config.EPOCHS)

# 最良のモデルをロードして予測を行う
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pt')))

val = pl.read_parquet(train_file_list[9], n_rows=n_rows).to_pandas()
_, _, X_val, y_val =prepare_dataloader(val, val, FEATURES, TARGETS, Config.DEVICE)
oof = predict_in_batches(model, X_val, Config.BATCH_SIZE)
print("Val score = ", util.get_score(y_val.cpu().numpy(), oof.detach().cpu().numpy()))

test_tensor = torch.tensor(test.values, dtype=torch.float32).to(Config.DEVICE)
preds = predict_in_batches(model, test_tensor, Config.BATCH_SIZE)
all_preds.append(preds)


# CVのアンサンブル
preds = np.mean(all_preds, axis=0)


# In[ ]:


# trainのスコア
train = pl.read_parquet(train_file_list[0], n_rows=n_rows).to_pandas()
targets = train[TARGETS].values
train_tensor = torch.tensor(train[FEATURES].values, dtype=torch.float32).to(Config.DEVICE)
train_preds = predict_in_batches(model, train_tensor, Config.BATCH_SIZE)
print("Train score = ", util.get_score(targets, train_preds.detach().cpu().numpy()))


# In[ ]:


# local testの予測と結果
local_test = pl.read_parquet(os.path.join(PROCESSED_DIR, 'local_test_enc.parquet'))
local_test = local_test.to_pandas()

target = local_test[TARGETS].values
local_test_tensor = torch.tensor(local_test[FEATURES].values, dtype=torch.float32).to(Config.DEVICE)
local_preds = predict_in_batches(model, local_test_tensor, Config.BATCH_SIZE)

print("Local test score = ", util.get_score(target, local_preds.detach().cpu().numpy()))





# テストデータの読み込み
tst = pl.read_parquet(os.path.join(RAW_DIR, "test.parquet"), n_rows=None).to_pandas()

# 'binds'列を追加して初期化
tst['binds'] = 0

# ブールマスクの作成
mask_BRD4 = (tst['protein_name'] == 'BRD4').values
mask_HSA = (tst['protein_name'] == 'HSA').values
mask_sEH = (tst['protein_name'] == 'sEH').values

# 各マスクに対応する予測値を代入
tst.loc[mask_BRD4, 'binds'] = preds[mask_BRD4][:, 0]
tst.loc[mask_HSA, 'binds'] = preds[mask_HSA][:, 1]
tst.loc[mask_sEH, 'binds'] = preds[mask_sEH][:, 2]



submission = tst[['id', 'binds']].copy()
# 'id'と'binds'列をCSVに出力
submission.to_csv(os.path.join(OUTPUT_DIR,'submission.csv'), index=False)


