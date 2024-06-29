# %%
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
import time
import sys
import datetime
import pytz
import matplotlib.pyplot as plt


from module import network, dataset, util
from importlib import reload

# %%
class Config:
    PREPROCESS = False
    KAGGLE_NOTEBOOK = False
    DEBUG = False
    MODEL = 'lstm'
    SEED = 42
    EPOCHS = 9*3
    BATCH_SIZE = 4096
    LR = 1e-4
    WD = 1e-6
    PATIENCE = 3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EARLY_STOPPING = False
    NUM_CV = 1
    VAL_INDEX = [1]
    NOTEBOOK = False
    LOAD_MODEL = True
    # models配下
    MODEL_PATH = "lstm/lstm_fold1_27.pt"
    
    
if Config.DEBUG:
    n_rows = 10**4
else:
    n_rows = None



# %%
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
    LOG_DIR = "../data/logs"
    LOSS_DIR = "../data/losses"

TRAIN_DATA_NAME = "train_enc.parquet"

# %%
tz_japan = pytz.timezone('Asia/Tokyo')
# 現在の日本時間を取得してログファイル名を生成
current_time = datetime.datetime.now(tz_japan).strftime("%Y_%m_%d_%H:%M:%S")
log_filename = os.path.join(LOG_DIR, f"{Config.MODEL}_{current_time}")

# %%
def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seeds(seed=Config.SEED)


    
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

# %%


class Trainer:
    def __init__(self, model, criterion, optimizer, device, patience, scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.scheduler = scheduler
        self.loss = {"train": [], "val": []}
        self.score = {"train": [], "val": []}

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        score_list = []
        for inputs, targets in train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            # caluculate the score(almost APS)
            score_list.append(util.get_score(targets.detach().cpu().numpy(), outputs.detach().cpu().numpy()))


        epoch_loss = running_loss / len(train_loader.dataset)
        
        self.loss["train"].append(epoch_loss)
        self.score["train"].append(np.mean(score_list))

    def validate(self, valid_loader):
        self.model.eval()
        val_loss = 0.0
        score_list = []
        with torch.no_grad():
            for inputs, targets in valid_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                score_list.append(util.get_score(targets.detach().cpu().numpy(), outputs.detach().cpu().numpy()))

        val_loss /= len(valid_loader.dataset)
        self.loss["val"].append(val_loss)
        self.score["val"].append(np.mean(score_list))
        # scheduler
        self.scheduler.step(val_loss)
        
    
    def train(self, train_file_list, epochs):
        best_val_loss = float('inf')
        patience_counter = 0
        # valを固定 
        val = pl.read_parquet(train_file_list[9], n_rows=n_rows).to_pandas()
        # print("loaded val data", val.shape, train_file_list[9])
        for epoch in range(epochs):
            start_time = time.time()
            train = pl.read_parquet(train_file_list[epoch % 9], n_rows=n_rows).to_pandas()
            # print("loaded train data", train.shape, train_file_list[epoch % 9])
            train_loader, valid_loader, X_val, y_val = prepare_dataloader(train, val, FEATURES, TARGETS, Config.DEVICE)
            
            self.train_epoch(train_loader)
            self.validate(valid_loader)

            current_lr = self.optimizer.param_groups[0]['lr']    
            print(f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {self.loss['train'][-1]:.4f}, Train Score: {self.score['train'][-1]:.4f} ,Val Loss: {self.loss['val'][-1]:.4f}, Val Score: {self.score['val'][-1]:.4f}, LR: {current_lr:.1e}")

            
            if Config.EARLY_STOPPING:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pt'))
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print('Early stopping')
                        break
            else :
                torch.save(self.model.state_dict(), os.path.join(MODEL_DIR, 'best_model.pt'))

            end_time = time.time()
            print(f"Time: {(end_time - start_time)//60} mins")
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

# %%


FEATURES = [f'enc{i}' for i in range(142)]
TARGETS = ['bind1', 'bind2', 'bind3']


def train_model():
    print(f"Config: {Config.__dict__}")
    print(n_rows)
    
    pos_weight = torch.tensor([215, 241, 136], device=Config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


    for i, val_index in enumerate(Config.VAL_INDEX):
        print(f"Cross Validation: {i+1}/{len(Config.VAL_INDEX)}")
        train_index = [index for index in range(10) if index not in [val_index]]
        train_index.append(val_index)
        train_file_list = [f"../data/chuncked-dataset/local_train_enc_{i}.parquet" for i in train_index]


        model = network.LSTMModel().to(Config.DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WD)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=Config.PATIENCE, min_lr=1e-6)
        
        model = torch.compile(model)
        if Config.LOAD_MODEL:
            model_path = os.path.join(MODEL_DIR, Config.MODEL_PATH)
            model.load_state_dict(torch.load(model_path))
            print(f"model loaded {model_path}")
        else :
            print("model initialized")
        
        
        
        trainer = Trainer(model, criterion, optimizer, Config.DEVICE, Config.PATIENCE, scheduler)
        trainer.train(train_file_list, Config.EPOCHS)
        
        # ロスの保存
        loss_path = os.path.join(LOSS_DIR, f'{Config.MODEL}_fold{val_index}_loss.npy')
        np.save(loss_path, trainer.loss)
        print("loss saved", loss_path)

        # 最良のモデルをロードして予測を行う
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pt')))

        val = pl.read_parquet(train_file_list[9], n_rows=n_rows).to_pandas()
        _, _, X_val, y_val =prepare_dataloader(val, val, FEATURES, TARGETS, Config.DEVICE)
        oof = predict_in_batches(model, X_val, Config.BATCH_SIZE)
        print(f"Val score = {util.get_score(y_val.cpu().numpy(), oof.detach().cpu().numpy()):4f}")

            
        # train score
        train = pl.read_parquet(train_file_list[0], n_rows=10**5).to_pandas()
        targets = train[TARGETS].values
        train_tensor = torch.tensor(train[FEATURES].values, dtype=torch.float32).to(Config.DEVICE)
        train_preds = predict_in_batches(model, train_tensor, Config.BATCH_SIZE)
        print(f"Train score = {util.get_score(targets, train_preds.detach().cpu().numpy()):.4f}")


        # local testの予測と結果
        local_test = pl.read_parquet(os.path.join(PROCESSED_DIR, 'local_test_enc.parquet'))
        local_test = local_test.to_pandas()

        target = local_test[TARGETS].values
        local_test_tensor = torch.tensor(local_test[FEATURES].values, dtype=torch.float32).to(Config.DEVICE)
        local_preds = predict_in_batches(model, local_test_tensor, Config.BATCH_SIZE)

        print(f"Local test score = {util.get_score(target, local_preds.detach().cpu().numpy()):.4f}")
        model_path = os.path.join(MODEL_DIR, f'{Config.MODEL}_fold{val_index}.pt')
        torch.save(model.state_dict(), model_path)
        print(f"model saved {model_path}")



# %%

def main():

    # 逐一出力
    with open(log_filename+".log", "w", buffering=1) as file:
        old_stdout = sys.stdout
        sys.stdout = file
        
        train_model()
        
        # stdoutを元に戻す
        sys.stdout = old_stdout
    
    

if Config.NOTEBOOK:
    train_model()
    
else:
    if __name__ == "__main__":
        main()

# %%



