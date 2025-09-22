import pickle
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import torch_geometric.transforms as T
import networkx as nx
import pandas as pd
import numpy as np
import torch


# data load
def load_obj( name ):
    """
    Load dataset from pickle file.
    :param name: Full pathname of the pickle file
    :return: Dataset type of dictionary
    """
    with open( name , 'rb') as f:
        return pickle.load(f)


def load_net_specific_data(args):
    """
    Load network-specific dataset from the pickle file.
    :param args: Arguments received from command line
    :return: Data for training model (class: 'torch_geometric.data.Data')
    """
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





