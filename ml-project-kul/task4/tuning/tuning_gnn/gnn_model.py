import os
import importlib.util
import sys
import math
from tqdm import tqdm
import csv
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.nn import GATConv, global_mean_pool

from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
from torch.utils.data import Dataset
from torch.optim import lr_scheduler

log = logging.getLogger(__name__)

from task4.training.training_gnn.arena import *
from task4.training.training_gnn.graph import *


class GNNetWrapper():
    def __init__(self, config, save_info=False):
        self.nnet = CustomGNN(num_node_features=5, num_edge_features=3, channels=config['num_channels'])
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    def to(self, device):
        self.nnet.to(device)
        return self
        
    def custom_loss(self, pi, target_pi, value, target_value):
        pi = target_pi.view_as(target_pi)
        mse_loss = nn.MSELoss()(value.view(-1),target_value.view(-1))
        cross_entropy_loss = nn.CrossEntropyLoss()(pi,target_pi)
        l2_reg = torch.tensor(0., device=value.device)
        for param in self.nnet.parameters():
            l2_reg += torch.norm(param,p=2)**2
        l2_loss = self.config['l2_coeff']* l2_reg

        return mse_loss, cross_entropy_loss, l2_loss
    
    def train(self, examples):
        input_graphs, target_pis, target_values = list(zip(*examples))

        
        self.nnet.to(self.device)
        
        optimizer = torch.optim.Adam(self.nnet.parameters(), lr=self.config['lr'])

        train_dataset = CustomGraphDataset(input_graphs,target_pis,target_values)
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, collate_fn=custom_collate)

        for epoch in tqdm(range(self.config['epocs_gnn']),desc="Training GNNet"):
            self.nnet.train()
            total_loss = 0
            total_value_loss = 0
            total_policy_loss = 0
            total_reg_loss = 0

            for graph,target_pi,target_value in train_loader:
                graph = graph.to(self.device)
                target_pi = target_pi.to(self.device)
                target_value = target_value.to(self.device)

                optimizer.zero_grad()
                pred_pi, pred_value = self.nnet(graph)

                value_loss , policy_loss, l2_loss = self.custom_loss(pred_pi,target_pi,pred_value,target_value)
                loss = value_loss + policy_loss + l2_loss

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_value_loss += value_loss.item()
                total_policy_loss += policy_loss.item()
                total_reg_loss += l2_loss.item()

    def predict(self, state):
        data = state_to_graph_data(state)
        self.nnet.to(self.device) 
        data = data.to(self.device)
        self.nnet.eval()
        with torch.no_grad():
            edge_probs,value = self.nnet(data)

        return edge_probs, value

    def load_checkpoint(self, folder, filename):
        full_path = os.path.join(folder, filename)
        if os.path.exists(full_path):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.nnet.load_state_dict(torch.load(full_path, map_location=device))
            log.info(f"Loaded weights from {full_path} to {device}")
        else:
            log.warning(f"No checkpoint found at {full_path}")
        
    def save_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            os.mkdir(folder)
        else:
            torch.save(self.nnet.state_dict(), filepath)
            log.info(f"Checkpoint saved at {filepath}")

    

class CustomGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, channels):
        super(CustomGNN, self).__init__()

        self.conv1 = GATConv(in_channels=num_node_features, out_channels=channels, heads=8, concat=False, edge_dim=num_edge_features)
        self.conv2 = GATConv(in_channels=channels, out_channels=channels, heads=8, concat=False, edge_dim=num_edge_features)
        self.conv3 = GATConv(in_channels=channels, out_channels=channels, heads=8, concat=False, edge_dim=num_edge_features)

        self.fc1 = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(), nn.BatchNorm1d(channels), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(), nn.BatchNorm1d(channels), nn.Dropout(0.5))

        # Policy and Value heads
        self.policy_head = nn.Sequential(nn.Linear(channels, 1), nn.Sigmoid())
        self.value_head = nn.Sequential(nn.Linear(channels, 1), nn.Tanh())

    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x1 = F.relu(self.conv1(x, edge_index, edge_attr))
        x2 = F.relu(self.conv2(x1, edge_index, edge_attr))
        x3 = F.relu(self.conv3(x2, edge_index, edge_attr))

        x_concat = torch.cat([x1, x2, x3], dim=-1)
        x_fc = self.fc2(self.fc1(x3))

        # Policy head predicts the probability of choosing each edge
        edge_probs = self.policy_head(x_fc[edge_index[0]]).squeeze()
        # Value head predicts the overall value of the state
        value = self.value_head(global_mean_pool(x_fc, batch)).squeeze()

        return edge_probs, value


class CustomGraphDataset(Dataset):
    def __init__(self, input_graphs, target_pis, target_values):
        assert all(isinstance(g, Data) for g in input_graphs), "All input_graphs should be torch_geometric.data.Data objects"
        self.input_graphs = input_graphs
        self.target_pis = target_pis
        self.target_values = target_values

    def __getitem__(self, index):
        return self.input_graphs[index], torch.tensor(self.target_pis[index]), torch.tensor(self.target_values[index])

    def __len__(self):
        return len(self.input_graphs)


def custom_collate(batch):
    input_graphs, target_pis, target_values = zip(*batch)
    batch_graph = Batch.from_data_list(list(input_graphs))  # Ensure input_graphs is a list of Data objects
    max_action_size = max(pi.shape[0] for pi in target_pis)  # Ensure correct shape handling
    batch_target_pis = torch.zeros((len(target_pis), max_action_size), dtype=torch.float)
    for i, pi in enumerate(target_pis):
        batch_target_pis[i, :pi.shape[0]] = torch.tensor(pi, dtype=torch.float)
    batch_target_values = torch.tensor(target_values, dtype=torch.float)
    return batch_graph, batch_target_pis, batch_target_values
