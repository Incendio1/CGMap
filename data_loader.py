import pickle
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import torch_geometric.transforms as T
import networkx as nx
import pandas as pd
import numpy as np
import torch


def load_obj( name ):
    with open( name , 'rb') as f:
        return pickle.load(f)


def load_net_specific_data(args):
    dataset_path = f"./data/{args.dataset}/dataset_{args.dataset}_ten_5CV.pkl"
    dataset = load_obj(dataset_path)
    std = StandardScaler()
    features = std.fit_transform(dataset['feature'].detach().numpy())
    features = torch.FloatTensor(features)
    if args.cross_validation:
        mask = dataset['split_set']
    else:
        mask = dataset['mask']
    data = Data(x=features, y=dataset['label'], edge_index=dataset['edge_index'], mask=mask, node_names=dataset['node_name'])
    return data