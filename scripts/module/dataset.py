
import torch
from torch.utils.data import DataLoader, TensorDataset

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