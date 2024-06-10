
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


from functools import partial



class Config:
    PREPROCESS = False
    KAGGLE_NOTEBOOK = False
    DEBUG = True
    
    SEED = 42
    EPOCHS = 9
    BATCH_SIZE = 4096
    LR = 1e-4
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


# In[4]:


def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seeds(seed=Config.SEED)

train_file_list = [f"../data/chuncked-dataset/local_train_enc_{i}.parquet" for i in range(10)]
mask_file_list = [f"../data/chuncked-dataset/local_train_mask_{i}.parquet" for i in range(10)]


# # Preprocessing

# In[5]:




class FlashAttentionTransformerEncoder(nn.Module):
    def __init__(
        self,
        dim_model,
        num_layers,
        num_heads=None,
        dim_feedforward=None,
        dropout=0.0,
        norm_first=False,
        activation=F.gelu,
        rotary_emb_dim=0,
    ):
        super().__init__()

        try:
            from flash_attn.bert_padding import pad_input, unpad_input
            from flash_attn.modules.block import Block
            from flash_attn.modules.mha import MHA
            from flash_attn.modules.mlp import Mlp
        except ImportError:
            raise ImportError('Please install flash_attn from https://github.com/Dao-AILab/flash-attention')
        
        self._pad_input = pad_input
        self._unpad_input = unpad_input

        if num_heads is None:
            num_heads = dim_model // 64
        
        if dim_feedforward is None:
            dim_feedforward = dim_model * 4

        if isinstance(activation, str):
            activation = {
                'relu': F.relu,
                'gelu': F.gelu
            }.get(activation)

            if activation is None:
                raise ValueError(f'Unknown activation {activation}')

        mixer_cls = partial(
            MHA,
            num_heads=num_heads,
            use_flash_attn=True,
            rotary_emb_dim=rotary_emb_dim
        )

        mlp_cls = partial(Mlp, hidden_features=dim_feedforward)

        self.layers = nn.ModuleList([
            Block(
                dim_model,
                mixer_cls=mixer_cls,
                mlp_cls=mlp_cls,
                resid_dropout1=dropout,
                resid_dropout2=dropout,
                prenorm=norm_first,
            ) for _ in range(num_layers)
        ])
    
    def forward(self, x, src_key_padding_mask=None):
        batch, seqlen = x.shape[:2]

        if src_key_padding_mask is None:
            for layer in self.layers:
                x = layer(x)
        else:
            x, indices, cu_seqlens, max_seqlen_in_batch = self._unpad_input(x, ~src_key_padding_mask)
            
            for layer in self.layers:
                x = layer(x, mixer_kwargs=dict(
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen_in_batch
                ))
      

            x = self._pad_input(x, indices, batch, seqlen)
            
        return x

class Conv1dBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, is_bn, **kwargs):
        super(Conv1dBnRelu, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.is_bn = is_bn
        if self.is_bn:
            self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.is_bn:
            x = self.bn(x)
        return self.relu(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[ :,:x.size(1)]
        
        return x

class Net(nn.Module):
    def __init__(self, ):
        super().__init__()

        embed_dim=512

        self.output_type = ['infer', 'loss']
        self.pe = PositionalEncoding(embed_dim,max_len=256)
        self.embedding = nn.Embedding(VOCAB_SIZE, 64, padding_idx=PAD)
        self.conv_embedding = nn.Sequential(
            Conv1dBnRelu(64, embed_dim, kernel_size=3,stride=1,padding=1, is_bn=True),
        )  #just a simple conv1d-bn-relu . for bn use: BN = partial(nn.BatchNorm1d, eps=5e-3,momentum=0.1)

        self.tx_encoder = FlashAttentionTransformerEncoder(
            dim_model=embed_dim,
            num_heads=8,
            dim_feedforward=embed_dim*4,
            dropout=0.1,
            norm_first=False,
            activation=F.gelu,
            rotary_emb_dim=0,
            num_layers=7,
        )

        self.bind = nn.Sequential(
            nn.Linear(embed_dim, 3),
        )


    def forward(self, batch):
        smiles_token_id   = batch['smiles_token_id'].long()
        smiles_token_mask = batch['smiles_token_mask'].long()
        B, L = smiles_token_id.shape
        x = self.embedding(smiles_token_id)
        x = x.permute(0,2,1).float()
        x = self.conv_embedding(x)
        x = x.permute(0,2,1).contiguous()

        x = self.pe(x)
        z = self.tx_encoder(
            x=x,
            src_key_padding_mask=smiles_token_mask==0,
        )


        m = smiles_token_mask.unsqueeze(-1).float()
        pool = (z*m).sum(1)/m.sum(1)
        bind = self.bind(pool)

        # --------------------------
        output = {}
        if 'loss' in self.output_type:
            target = batch['bind']
            output['bce_loss'] = F.binary_cross_entropy_with_logits(bind.float(), target.float())

        if 'infer' in self.output_type:
            output['bind'] = torch.sigmoid(bind)

        return output

    



class Trainer:
    def __init__(self, model, optimizer, device, patience):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.patience = patience

    def train_epoch(self, train_df, mask_df):
        self.model.train()
        running_loss = 0.0
        total_samples = 0
        "TODO: バッチじゃなくてdata loaderを使うべき．データサイズが大きいと遅い"
        for batch in get_batch(train_df, mask_df, batch_size=4096):
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                output = self.model(batch)
            loss = output['bce_loss']  # モデル内で計算される損失
            loss.backward()
            self.optimizer.step()
            batch_size = batch['smiles_token_id'].size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        epoch_loss = running_loss / total_samples
        return epoch_loss

    def validate(self, valid_df, mask_df):
        self.model.eval()
        val_loss = 0.0
        total_samples = 0
        "TODO: バッチじゃなくてdata loaderのほうが効率いいかも"
        with torch.no_grad():
            for batch in get_batch(valid_df, mask_df, batch_size=4096):
                with torch.cuda.amp.autocast(enabled=True):
                    output = self.model(batch)
                loss = output['bce_loss']
                batch_size = batch['smiles_token_id'].size(0)
                val_loss += loss.item() * batch_size
                total_samples += batch_size

        val_loss /= total_samples
        return val_loss

    def train(self, train_file_list, mask_file_list, epochs):
        best_val_loss = float('inf')
        patience_counter = 0

        
        for epoch in range(epochs):
            train_df = pl.read_parquet(train_file_list[epoch % 9], n_rows=n_rows).to_pandas()
            mask_df = pl.read_parquet(mask_file_list[epoch % 9], n_rows=n_rows).to_pandas()
            
    
            epoch_loss = self.train_epoch(train_df, mask_df)
            del train_df, mask_df
            gc.collect()
            val_df = pl.read_parquet(train_file_list[9], n_rows=n_rows).to_pandas()
            val_mask_df = pl.read_parquet(mask_file_list[9], n_rows=n_rows).to_pandas()
            val_loss = self.validate(val_df, val_mask_df)  
            del val_df, val_mask_df
            gc.collect()

            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

            if Config.EARLY_STOPPING:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), os.path.join(MODEL_DIR, f'best_model.pt'))
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print('早期終了')
                        break
            else:
                torch.save(self.model.state_dict(), os.path.join(MODEL_DIR, f'best_model.pt'))
                print("model saved")
               

        return best_val_loss

def get_batch(df, mask_df, batch_size=32):
    for i in range(0, len(df), batch_size):
        batch = {
            'smiles_token_id': torch.from_numpy(df[FEATURES].values[i:i + batch_size]).byte().to(Config.DEVICE),
            'smiles_token_mask': torch.from_numpy(mask_df[FEATURES].values[i:i + batch_size]).byte().to(Config.DEVICE),
            'bind': torch.from_numpy(df[TARGETS].values[i:i + batch_size]).float().to(Config.DEVICE)
        }
        yield batch


def predict_in_batches(model, val_df, val_mask_df, batch_size=4096):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in get_batch(val_df, val_mask_df, batch_size=4096):
            with torch.cuda.amp.autocast(enabled=True):
                output = model(batch)
            output = output["bind"]
            preds.append(output)
    preds = torch.cat(preds, dim=0).cpu().numpy()
    return preds



# 定数やモデルの定義は適宜修正してください
FEATURES = [f'enc{i}' for i in range(142)]
TARGETS = ['bind1', 'bind2', 'bind3']


# https://www.ascii-code.com/
MOLECULE_DICT = {
    'l': 1, 'y': 2, '@': 3, '3': 4, 'H': 5, 'S': 6, 'F': 7, 'C': 8, 'r': 9, 's': 10, '/': 11, 'c': 12, 'o': 13,
    '+': 14, 'I': 15, '5': 16, '(': 17, '2': 18, ')': 19, '9': 20, 'i': 21, '#': 22, '6': 23, '8': 24, '4': 25,
    '=': 26, '1': 27, 'O': 28, '[': 29, 'D': 30, 'B': 31, ']': 32, 'N': 33, '7': 34, 'n': 35, '-': 36
}
MAX_MOLECULE_ID = np.max(list(MOLECULE_DICT.values()))
VOCAB_SIZE = MAX_MOLECULE_ID + 10
UNK = 255  # disallow: will cuase error
BOS = MAX_MOLECULE_ID + 1
EOS = MAX_MOLECULE_ID + 2
# rest are reserved
PAD = 0
MAX_LENGTH = 160
model = Net().to(Config.DEVICE)
optimizer = optim.Adam(model.parameters(), lr=Config.LR, weight_decay=Config.WD)

# データの準備
trainer = Trainer(model, optimizer, Config.DEVICE, Config.PATIENCE)
trainer.train(train_file_list, mask_file_list, Config.EPOCHS)

# 最良のモデルをロードして予測を行う
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'best_model.pt')))


val_df = pl.read_parquet(train_file_list[9], n_rows=n_rows).to_pandas()
val_mask_df = pl.read_parquet(mask_file_list[9], n_rows=n_rows).to_pandas()

preds = predict_in_batches(model, val_df, val_mask_df)
print(preds)
val_score = util.get_score(val_df[TARGETS].values, preds)
print(f'Val Score: {val_score:.4f}')


train_df = pl.read_parquet(train_file_list[0], n_rows=n_rows).to_pandas()
train_mask_df = pl.read_parquet(mask_file_list[0], n_rows=n_rows).to_pandas()
preds = predict_in_batches(model, train_df, train_mask_df)
print(preds)
train_score = util.get_score(train_df[TARGETS].values, preds)
print(f'train Score: {train_score:.4f}')

