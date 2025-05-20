"""
Author: Malladi Tejasvi (CS23MO36), M.Tech CSE, IIT Madras.
Created: 8 September 2024

Dataset to CSR Converter for Graph Neural Network Tasks
===============================================================================

This script processes various graph datasets from PyTorch Geometric and OGB libraries 
into standard formats usable by C++ implementations of Graph Neural Networks. It converts 
graph topology information into Compressed Sparse Row (CSR) format and exports node 
features, labels, and train/val/test splits.

Supported Datasets:
------------------
- Citation networks: Cora, CiteSeer, PubMed
- Social networks: Reddit, Flickr
- E-commerce: Amazon-Computers, Amazon-Photo, Yelp
- OGB datasets: ogbn-products, ogbn-arxiv, ogbn-papers100M

Features:
---------
- Converts graph data to edge lists and CSR format
- Handles different dataset splits (public, full, random)
- Optional feature normalization
- Exports train/val/test indices
- Preserves node features and labels

Usage:
------
    python JV_Dataset2CSR.py [-s SPLIT] [-n NORM] [-ds DATASET]

Arguments:
    -s, --split    Split type: "public", "full", or "random" (default: "public")
    -n, --norm     Whether to normalize features: True or False (default: False)
    -ds, --dataset Dataset name (default: "PubMed")
                   Choices: ["PubMed", "Cora", "CiteSeer", "Reddit", "Yelp", "Nell", 
                             "Amazon-Computers", "Amazon-Photo", "Flickr", "products", 
                             "arxiv", "papers100M"]

Example:
--------
    python JV_Dataset2CSR.py --dataset Cora --split public --norm True

Requirements:
------------
Python: 3.8
torch==2.1.2 
torch_geometric==2.3.1
"""


import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_networkx, from_networkx
import torch_geometric.datasets as tg_datasets
from torch_geometric.datasets import Flickr
from torch_geometric.datasets import NELL
from torch_geometric.datasets import Yelp
from torch_geometric.utils import index_to_mask
import networkx as nx
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import DataLoader

import time

import argparse


parser = argparse.ArgumentParser()

parser.add_argument("-s", "--split", type=str, default="public",help="Type of splitting.",choices=["public","full","random"])
parser.add_argument("-n", "--norm", type=bool, default=False,help="Whether features should be Normalized.",choices=[True,False])
parser.add_argument("-ds", "--dataset", type=str, default="PubMed",help="Name of the required dataset.",choices=["PubMed","Cora","CiteSeer","Reddit","Yelp","Nell","Amazon-Computers","Amazon-Photo","Flickr","products","arxiv","papers100M"])
args = parser.parse_args()

# In[3]:

is_ogbn = False

if args.dataset == "products" or args.dataset == "arxiv" or args.dataset == "papers100M":
    
    is_ogbn = True
    
    dataset = PygNodePropPredDataset(name='ogbn-'+args.dataset,root='Datasets/ogbn-'+args.dataset)
    data = dataset[0]
    
    split_idx = dataset.get_idx_split()
    
    # if args.dataset == "mag":
    #     split_idx["train"] = split_idx["train"]["paper"]
    #     split_idx["valid"] = split_idx["valid"]["paper"]
    #     split_idx["test"] = split_idx["test"]["paper"]


    if 'num_nodes_dict' in data:
        data.num_nodes = sum(data.num_nodes_dict.values())
        
    train_mask = index_to_mask(split_idx["train"], size=data.num_nodes)
    val_mask = index_to_mask(split_idx["valid"], size=data.num_nodes)
    
    test_mask = index_to_mask(split_idx["test"], size=data.num_nodes)


if args.dataset == "Reddit":
    if args.norm:
        dataset = tg_datasets.Reddit(root='Datasets/'+args.dataset+"/",transform=NormalizeFeatures(),force_reload=False)
    else:
        dataset = tg_datasets.Reddit(root='Datasets/'+args.dataset+"/",force_reload=False)

elif args.dataset == "Nell":
    if args.norm:
        dataset = tg_datasets.NELL(root='Datasets/'+args.dataset+"/",transform=NormalizeFeatures(),force_reload=False)
    else:
        dataset = tg_datasets.NELL(root='Datasets/'+args.dataset+"/",force_reload=False)

elif args.dataset == "Yelp":
    if args.norm:
        dataset = tg_datasets.Yelp(root='Datasets/'+args.dataset+"/",transform=NormalizeFeatures(),force_reload=False)
    else:
        dataset = tg_datasets.Yelp(root='Datasets/'+args.dataset+"/",force_reload=False)

elif args.dataset == "Amazon-Computers":
    if args.norm:
        dataset = tg_datasets.Amazon(root='Datasets/'+args.dataset+"/",name="Computers",transform=NormalizeFeatures(),force_reload=False)
    else:
        dataset = tg_datasets.Amazon(root='Datasets/'+args.dataset+"/",name="Computers",force_reload=False)

elif args.dataset == "Amazon-Photo":
    if args.norm:
        dataset = tg_datasets.Amazon(root='Datasets/'+args.dataset+"/",name="Photo",transform=NormalizeFeatures(),force_reload=False)
    else:
        dataset = tg_datasets.Amazon(root='Datasets/'+args.dataset+"/",name="Photo",force_reload=False)

if args.dataset == "Flickr":
    if args.norm:
        dataset = Flickr(root='Datasets/Flickr',transform=NormalizeFeatures(),force_reload=False)
    else:
        dataset = Flickr(root='Datasets/Flickr',force_reload=False)

elif args.dataset == "Cora" or args.dataset == "CiteSeer" or args.dataset == "PubMed":
    if args.norm:
        dataset = Planetoid(root='Datasets/'+args.dataset+"/", name=args.dataset,split=args.split, transform=NormalizeFeatures(),force_reload=False)
    else:
        dataset = Planetoid(root='Datasets/'+args.dataset+"/", name=args.dataset,split=args.split,force_reload=False)


if "Amazon" in args.dataset:

    data = dataset[0]

    # Get the number of nodes in the dataset
    num_nodes = data.num_nodes

    # Create random indices for train, validation, and test
    torch.manual_seed(42)  # For reproducibility

    # Split percentages
    train_size = int(0.7 * num_nodes)
    val_size = int(0.1 * num_nodes)
    test_size = num_nodes - train_size - val_size

    # Random permutation of indices
    indices = torch.randperm(num_nodes)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    # Create masks for training, validation, and testing
    train_mask = index_to_mask(train_idx, size=num_nodes)
    val_mask = index_to_mask(val_idx, size=num_nodes)
    test_mask = index_to_mask(test_idx, size=num_nodes)


elif not is_ogbn:
    # Get the data object
    data = dataset[0]

    # Train, validation, and test masks
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

train_indices = train_mask.nonzero(as_tuple=True)[0]
val_indices = val_mask.nonzero(as_tuple=True)[0]
test_indices = test_mask.nonzero(as_tuple=True)[0]


# In[4]:


#data.x.shape,data.edge_index.shape


# In[5]:


#Adjacency list of PubMed Graph as a py dict
adj_list = {}


"""
Inductive Learning : Neighbours of a vertex in train set, are considered for sampling even if they are not in train set.

Hence constructing CSR of the entire graph, from the adjacency list constructed below:

"""

# Initialize adjacency list for all nodes
for node in range(data.num_nodes):
    adj_list[node] = []

for e in range(data.edge_index.shape[1]):
    u = data.edge_index[0][e].item()
    v = data.edge_index[1][e].item()

    if u not in adj_list:
        adj_list[u] = []
    adj_list[u].append(v)

    
## CSR from Adj List
index_list = []
CSR = []
pos = 0
for i in range(len(adj_list)):
    index_list.append(pos)
    CSR += adj_list[i]
    pos += len(adj_list[i])
    

#Converting CSR and vertex features into numpy arrays:
CSR = np.array(CSR).astype(int)
index_list = np.array(index_list).astype(int)
features = data.x.numpy()
train_indices = train_indices.numpy().astype(int)
val_indices = val_indices.numpy().astype(int)
test_indices = test_indices.numpy().astype(int)
labels = data.y.numpy().astype(int)

data_dir = 'Datasets/'+args.dataset+"/"

# #Saving them locally:
#np.savetxt('CSR.txt', CSR, delimiter=' ',fmt="%i")
#np.savetxt('index.txt', index_list, delimiter=' ',fmt="%i")
np.savetxt('edgelist.txt', data.edge_index.t(), delimiter=' ',fmt="%i")
np.savetxt('features.txt', features, delimiter=' ')
np.savetxt('train_indices.txt', train_indices, delimiter=' ',fmt="%i")
np.savetxt('val_indices.txt', val_indices, delimiter=' ',fmt="%i")
np.savetxt('test_indices.txt', test_indices, delimiter=' ',fmt="%i")
np.savetxt('labels.txt', labels, delimiter=' ',fmt="%i")