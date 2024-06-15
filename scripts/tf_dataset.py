import numpy as np
import polars as pl
import pandas as pd
import os

import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import math
from functools import partial
import time

class Config:
    PREPROCESS = False
    KAGGLE_NOTEBOOK = False
    DEBUG = False
    
    SEED = 42
    EPOCHS = 10
    BATCH_SIZE = 4096
    LR = 1e-3
    WD = 1e-6
    PATIENCE = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NBR_FOLDS = 15
    SELECTED_FOLDS = [0]
    
    
if Config.DEBUG:
    n_rows = 10**5
else:
    n_rows = None
    
print(Config.__dict__, n_rows)
    

if Config.KAGGLE_NOTEBOOK:
    RAW_DIR = "/kaggle/input/leash-BELKA/"
    PROCESSED_DIR = "/kaggle/input/belka-enc-dataset"
    OUTPUT_DIR = ""
    MODEL_DIR = ""
else:
    RAW_DIR = "../data/raw/"
    PROCESSED_DIR = "../data/processed/"
    OUTPUT_DIR = "../data/tf-dataset/"
    MODEL_DIR = "../models/"

TRAIN_DATA_NAME = "train_enc.parquet"


def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seeds(seed=Config.SEED)



#tokenization ====================================

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

MOLECULE_LUT = np.full(256, fill_value=UNK, dtype=np.uint8)
for k, v in MOLECULE_DICT.items():
    ascii = ord(k)
    MOLECULE_LUT[ascii] = v


# SMILESトークン化関数
def make_token(s):
    MOLECULE_LUT = np.full(256, fill_value=255, dtype=np.uint8)
    for k, v in MOLECULE_DICT.items():
        MOLECULE_LUT[ord(k)] = v
    t = np.frombuffer(s.encode(), np.uint8)
    t = MOLECULE_LUT[t]
    t = t.tolist()
    L = len(t) + 2
    token_id = [37] + t + [38] + [0] * (160 - L)
    token_mask = [1] * L + [0] * (160 - L)
    return token_id, token_mask


# トークン列とマスク列をデータフレームに変換
def expand_tokens_to_df(tokens, max_length):
    token_columns = {f"enc{i}": [] for i in range(max_length)}
    mask_columns = {f"enc{i}": [] for i in range(max_length)}

    for token_id, mask in tokens:
        for i in range(max_length):
            token_columns[f"enc{i}"].append(token_id[i] if i < len(token_id) else None)
            mask_columns[f"enc{i}"].append(mask[i] if i < len(mask) else None)

    token_df = pd.DataFrame(token_columns)
    mask_df = pd.DataFrame(mask_columns)

    return token_df, mask_df


# molecule_smilesのみを取得
for i in range(10):
    start_time = time.time()
    input_path = os.path.join("../data/shuffled-dataset/", f"train_{i}.parquet")
    train_raw = pl.read_parquet(input_path, n_rows=n_rows, columns=["molecule_smiles", "bind1", "bind2", "bind3"]).to_pandas()
    print("data loaded", input_path, train_raw.shape)
    smiles = train_raw["molecule_smiles"]
    tokens = smiles.apply(make_token)
    train, mask_df = expand_tokens_to_df(tokens, 160)
    train["bind1"] = train_raw["bind1"]
    train["bind2"] = train_raw["bind2"]
    train["bind3"] = train_raw["bind3"]

    # save
    path = os.path.join(OUTPUT_DIR, f"train_enc_{i}.parquet")
    train.to_parquet(path)
    mask_df.to_parquet(os.path.join(OUTPUT_DIR, f"train_mask_{i}.parquet"))
    print("data saved", path)
    end_time = time.time()
    print("time", int((end_time - start_time)/60), "min")


