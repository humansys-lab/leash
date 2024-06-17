#!/usr/bin/env python
# coding: utf-8

# for details, refer to https://www.kaggle.com/competitions/leash-BELKA/discussion/498858

# In[1]:


import numpy as np

import rdkit
from rdkit import Chem
import os
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_scatter import scatter
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score as APS
import pickle
import gzip
from multiprocessing import Pool


import pandas as pd
import torch
from torch_geometric.data import Data
import json
import time




# In[10]:




class Config:
    PREPROCESS = False
    KAGGLE_NOTEBOOK = False
    DEBUG = False
    
    SEED = 42
    EPOCHS = 4
    BATCH_SIZE = 4096
    LR = 1e-3
    WD = 1e-6*0
    PATIENCE = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NBR_FOLDS = 15
    SELECTED_FOLDS = [0]
    EARLY_STOPPING = False
    
    
if Config.DEBUG:
    n_rows = 10**4
else:
    n_rows = None
    
print(Config.__dict__, n_rows)


# In[3]:


torch.cuda.is_available()


# In[4]:


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


# In[5]:


# helper
# torch version of np unpackbits
#https://gist.github.com/vadimkantorov/30ea6d278bc492abf6ad328c6965613a

def tensor_dim_slice(tensor, dim, dim_slice):
	return tensor[(dim if dim >= 0 else dim + tensor.dim()) * (slice(None),) + (dim_slice,)]

# @torch.jit.script
def packshape(shape, dim: int = -1, mask: int = 0b00000001, dtype=torch.uint8, pack=True):
	dim = dim if dim >= 0 else dim + len(shape)
	bits, nibble = (
		8 if dtype is torch.uint8 else 16 if dtype is torch.int16 else 32 if dtype is torch.int32 else 64 if dtype is torch.int64 else 0), (
		1 if mask == 0b00000001 else 2 if mask == 0b00000011 else 4 if mask == 0b00001111 else 8 if mask == 0b11111111 else 0)
	# bits = torch.iinfo(dtype).bits # does not JIT compile
	assert nibble <= bits and bits % nibble == 0
	nibbles = bits // nibble
	shape = (shape[:dim] + (int(math.ceil(shape[dim] / nibbles)),) + shape[1 + dim:]) if pack else (
				shape[:dim] + (shape[dim] * nibbles,) + shape[1 + dim:])
	return shape, nibbles, nibble

# @torch.jit.script
def F_unpackbits(tensor, dim: int = -1, mask: int = 0b00000001, shape=None, out=None, dtype=torch.uint8):
	dim = dim if dim >= 0 else dim + tensor.dim()
	shape_, nibbles, nibble = packshape(tensor.shape, dim=dim, mask=mask, dtype=tensor.dtype, pack=False)
	shape = shape if shape is not None else shape_
	out = out if out is not None else torch.empty(shape, device=tensor.device, dtype=dtype)
	assert out.shape == shape

	if shape[dim] % nibbles == 0:
		shift = torch.arange((nibbles - 1) * nibble, -1, -nibble, dtype=torch.uint8, device=tensor.device)
		shift = shift.view(nibbles, *((1,) * (tensor.dim() - dim - 1)))
		return torch.bitwise_and((tensor.unsqueeze(1 + dim) >> shift).view_as(out), mask, out=out)

	else:
		for i in range(nibbles):
			shift = nibble * i
			sliced_output = tensor_dim_slice(out, dim, slice(i, None, nibbles))
			sliced_input = tensor.narrow(dim, 0, sliced_output.shape[dim])
			torch.bitwise_and(sliced_input >> shift, mask, out=sliced_output)
	return out

class dotdict(dict):
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__
	
	def __getattr__(self, name):
		try:
			return self[name]
		except KeyError:
			raise AttributeError(name)

            
print('helper ok!')


# In[6]:


# mol to graph adopted from
# from https://github.com/LiZhang30/GPCNDTA/blob/main/utils/DrugGraph.py

PACK_NODE_DIM=9
PACK_EDGE_DIM=1
NODE_DIM=PACK_NODE_DIM*8
EDGE_DIM=PACK_EDGE_DIM*8

def one_of_k_encoding(x, allowable_set, allow_unk=False):
	if x not in allowable_set:
		if allow_unk:
			x = allowable_set[-1]
		else:
			raise Exception(f'input {x} not in allowable set{allowable_set}!!!')
	return list(map(lambda s: x == s, allowable_set))


#Get features of an atom (one-hot encoding:)
'''
	1.atom element: 44+1 dimensions    
	2.the atom's hybridization: 5 dimensions
	3.degree of atom: 6 dimensions                        
	4.total number of H bound to atom: 6 dimensions
	5.number of implicit H bound to atom: 6 dimensions    
	6.whether the atom is on ring: 1 dimension
	7.whether the atom is aromatic: 1 dimension           
	Total: 70 dimensions
'''

ATOM_SYMBOL = [
	'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg',
	'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
	'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
	'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
	'Pt', 'Hg', 'Pb', 'Dy',
	#'Unknown'
]
#print('ATOM_SYMBOL', len(ATOM_SYMBOL))44
HYBRIDIZATION_TYPE = [
	Chem.rdchem.HybridizationType.S,
	Chem.rdchem.HybridizationType.SP,
	Chem.rdchem.HybridizationType.SP2,
	Chem.rdchem.HybridizationType.SP3,
	Chem.rdchem.HybridizationType.SP3D
]

def get_atom_feature(atom):
	feature = (
		 one_of_k_encoding(atom.GetSymbol(), ATOM_SYMBOL)
	   + one_of_k_encoding(atom.GetHybridization(), HYBRIDIZATION_TYPE)
	   + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
	   + one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5])
	   + one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])
	   + [atom.IsInRing()]
	   + [atom.GetIsAromatic()]
	)
	#feature = np.array(feature, dtype=np.uint8)
	feature = np.packbits(feature)
	return feature


#Get features of an edge (one-hot encoding)
'''
	1.single/double/triple/aromatic: 4 dimensions       
	2.the atom's hybridization: 1 dimensions
	3.whether the bond is on ring: 1 dimension          
	Total: 6 dimensions
'''

def get_bond_feature(bond):
	bond_type = bond.GetBondType()
	feature = [
		bond_type == Chem.rdchem.BondType.SINGLE,
		bond_type == Chem.rdchem.BondType.DOUBLE,
		bond_type == Chem.rdchem.BondType.TRIPLE,
		bond_type == Chem.rdchem.BondType.AROMATIC,
		bond.GetIsConjugated(),
		bond.IsInRing()
	]
	#feature = np.array(feature, dtype=np.uint8)
	feature = np.packbits(feature)
	return feature


def smile_to_graph(smiles):
	mol = Chem.MolFromSmiles(smiles)
	N = mol.GetNumAtoms()
	node_feature = []
	edge_feature = []
	edge = []
	for i in range(mol.GetNumAtoms()):
		atom_i = mol.GetAtomWithIdx(i)
		atom_i_features = get_atom_feature(atom_i)
		node_feature.append(atom_i_features)

		for j in range(mol.GetNumAtoms()):
			bond_ij = mol.GetBondBetweenAtoms(i, j)
			if bond_ij is not None:
				edge.append([i, j])
				bond_features_ij = get_bond_feature(bond_ij)
				edge_feature.append(bond_features_ij)
	node_feature=np.stack(node_feature)
	edge_feature=np.stack(edge_feature)
	edge = np.array(edge,dtype=np.uint8)
	return N,edge,node_feature,edge_feature

def to_pyg_format(N,edge,node_feature,edge_feature):
	graph = Data(
		idx=-1,
		edge_index = torch.from_numpy(edge.T).int(),
		x          = torch.from_numpy(node_feature).byte(),
		edge_attr  = torch.from_numpy(edge_feature).byte(),
	)
	return graph

#debug one example
g = to_pyg_format(*smile_to_graph(smiles="C#CCOc1ccc(CNc2nc(NCc3cccc(Br)n3)nc(N[C@@H](CC#C)CC(=O)NC)n2)cc1"))
print(g)
print('[Dy] is replaced by C !!')
print('smile_to_graph() ok!')


# In[7]:


import json

from multiprocessing import Pool
from tqdm import tqdm
import gc
from torch_geometric.loader import DataLoader as PyGDataLoader

def to_pyg_list(graph):
	L = len(graph)
	for i in tqdm(range(L)):
		N, edge, node_feature, edge_feature = graph[i]
		graph[i] = Data(
			idx=i,
			edge_index=torch.from_numpy(edge.T).int(),
			x=torch.from_numpy(node_feature).byte(),
			edge_attr=torch.from_numpy(edge_feature).byte(),
		)
	return graph


# In[8]:




INPUT_DIR = "../data/shuffled-dataset/"
SAVE_PATH = "../data/graph/"
for i in range(10):
    start = time.time()
    input_path = os.path.join(INPUT_DIR, f"train_{i}.parquet")
    train = pl.read_parquet(input_path, n_rows=n_rows, columns=["molecule_smiles"]).to_pandas()
    print("data loaded", input_path, train.shape)

    # グラフに変換
    train_smiles=train['molecule_smiles'].values
    num_train= len(train_smiles)
    with Pool(processes=64) as pool:
        train_graph = list(tqdm(pool.imap(smile_to_graph, train_smiles), total=num_train))
    train_graph = to_pyg_list(train_graph)
    output_path = os.path.join(SAVE_PATH, f"train_graph_{i}.parquet")
    torch.save(train_graph, output_path)
    print("data saved", output_path, len(train_graph))
    end = time.time()
    print("time", int((end - start)/60), "min")

