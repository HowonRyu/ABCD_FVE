import torch
from torch_geometric.nn import GCNConv, global_mean_pool, TopKPooling
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from FVE_GCN_utils import *
import glob
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


from distutils.util import strtobool
import argparse


#from torch_geometric.nn import TopKPooling
#from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
#from torch_geometric.utils import (add_self_loops, sort_edge_index, remove_self_loops)
#from torch_sparse import spspmm
#from net.braingraphconv import MyNNConv


if None:   # from BrainGNN Li et al. -- no longer use
    class Network(torch.nn.Module):
        def __init__(self, indim, ratio, nclass=1, k=8, R=3):
            '''
            :param indim: (int) node feature dimension
            :param ratio: (float) pooling ratio in (0,1)
            :param nclass: (int)  number of classes
            :param k: (int) number of communities
            :param R: (int) number of ROIs
            '''
            super(Network, self).__init__()

            self.indim = indim
            self.dim1 = 32
            self.dim2 = 32
            self.dim3 = 512
            self.dim4 = 256
            self.dim5 = 8
            self.k = k
            self.R = R

            #self.n1 = nn.Sequential(nn.Linear(self.indim, self.dim1),nn.ReLU(), nn.Linear(self.dim1, self.dim1))
            self.n1 = nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(), nn.Linear(self.k, self.dim1 * self.indim))
            self.conv1 = MyNNConv(self.indim, self.dim1, self.n1, normalize=False)
            self.pool1 = TopKPooling(self.dim1, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
            self.n2 = nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(), nn.Linear(self.k, self.dim2 * self.dim1))
            self.conv2 = MyNNConv(self.dim1, self.dim2, self.n2, normalize=False)
            self.pool2 = TopKPooling(self.dim2, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)

            #self.fc1 = torch.nn.Linear((self.dim2) * 2, self.dim2)
            #self.fc1 = torch.nn.Linear((self.dim1+self.dim2)*2, self.dim2)
            self.fc1 = torch.nn.Linear(2 * (self.dim1 + self.dim2), self.dim2)
            self.bn1 = torch.nn.BatchNorm1d(self.dim2)
            self.fc2 = torch.nn.Linear(self.dim2, self.dim3)
            self.bn2 = torch.nn.BatchNorm1d(self.dim3)
            self.fc3 = torch.nn.Linear(self.dim3, nclass)



        def forward(self, x, edge_index, batch, edge_attr, pos):
            x = self.conv1(x=x, edge_index=edge_index, edge_weight=edge_attr, pseudo=pos)
            x, edge_index, edge_attr, batch, perm, score1 = self.pool1(x, edge_index, edge_attr, batch)

            pos = pos[perm]
            x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            edge_attr = edge_attr.squeeze()
            edge_index, edge_attr = self.augment_adj(edge_index, edge_attr, x.size(0))

            x = self.conv2(x, edge_index, edge_attr, pos)
            x, edge_index, edge_attr, batch, perm, score2 = self.pool2(x, edge_index,edge_attr, batch)

            x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            x = torch.cat([x1,x2], dim=1)
            x = self.bn1(F.relu(self.fc1(x)))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.bn2(F.relu(self.fc2(x)))
            x= F.dropout(x, p=0.5, training=self.training)
            x = torch.sigmoid(self.fc3(x))  ### softmax eliminate, force to [0,1]

            #return x, self.pool1.weight, self.pool2.weight, torch.sigmoid(score1).view(x.size(0),-1), torch.sigmoid(score2).view(x.size(0),-1)
            return x

        def augment_adj(self, edge_index, edge_weight, num_nodes):
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                    num_nodes=num_nodes)
            edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                    num_nodes)
            edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                            edge_weight, num_nodes, num_nodes,
                                            num_nodes)
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
            return edge_index, edge_weight






class GCN_new(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim1=521, hidden_dim2=256, hidden_dim3=128, hidden_dim4=64, pool_ratio=0.2):
        super(GCN_new, self).__init__()
        
        # GCN + pooling layers
        self.conv1 = GCNConv(num_node_features, hidden_dim1, bias=True)
        self.pool1 = TopKPooling(hidden_dim1, ratio=pool_ratio)

        self.conv2 = GCNConv(hidden_dim1, hidden_dim2, bias=True)
        self.pool2 = TopKPooling(hidden_dim2, ratio=pool_ratio)

        self.conv3 = GCNConv(hidden_dim2, hidden_dim3, bias=True)
        self.pool3 = TopKPooling(hidden_dim3, ratio=pool_ratio)

        self.conv4 = GCNConv(hidden_dim3, hidden_dim4, bias=True)
        self.pool4 = TopKPooling(hidden_dim4, ratio=pool_ratio)

        # classifier after readout for regression
        self.linear = torch.nn.Linear(hidden_dim4 * 2, 1)

        # initialization
        torch.nn.init.kaiming_uniform_(self.conv1.lin.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv2.lin.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv3.lin.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv4.lin.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x, edge_index, batch, edge_attr=None):
        if edge_attr is not None:
            edge_weight = edge_attr.squeeze()
        else:
            edge_weight = None
            
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x, edge_index, edge_weight, batch, _, _ = self.pool1(x, edge_index, edge_weight, batch)

        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x, edge_index, edge_weight, batch, _, _ = self.pool2(x, edge_index, edge_weight, batch)

        x = F.relu(self.conv3(x, edge_index, edge_weight=edge_weight))
        x, edge_index, edge_weight, batch, _, _ = self.pool3(x, edge_index, edge_weight, batch)

        x = F.relu(self.conv4(x, edge_index, edge_weight=edge_weight))
        x, edge_index, edge_weight, batch, _, _ = self.pool4(x, edge_index, edge_weight, batch)

        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        return self.linear(x)



class GCN_new2(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim1=256, hidden_dim2=128, hidden_dim3=64, hidden_dim4=32, pool_ratio=0.6):
        super(GCN_new2, self).__init__()
        
        # GCN + pooling layers
        self.conv1 = GCNConv(num_node_features, hidden_dim1, bias=True)
        self.pool1 = TopKPooling(hidden_dim1, ratio=pool_ratio)

        self.conv2 = GCNConv(hidden_dim1, hidden_dim2, bias=True)
        self.pool2 = TopKPooling(hidden_dim2, ratio=pool_ratio)

        self.conv3 = GCNConv(hidden_dim2, hidden_dim3, bias=True)
        self.pool3 = TopKPooling(hidden_dim3, ratio=pool_ratio)

        self.conv4 = GCNConv(hidden_dim3, hidden_dim4, bias=True)
        self.pool4 = TopKPooling(hidden_dim4, ratio=pool_ratio)

        # classifier after readout for regression
        self.linear = torch.nn.Linear(hidden_dim4 * 2, 1)

        # initialization
        torch.nn.init.kaiming_uniform_(self.conv1.lin.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv2.lin.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv3.lin.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv4.lin.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x, edge_index, batch, edge_attr=None):
        if edge_attr is not None:
            edge_weight = edge_attr.squeeze()
        else:
            edge_weight = None
            
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x, edge_index, edge_weight, batch, _, _ = self.pool1(x, edge_index, edge_weight, batch)

        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x, edge_index, edge_weight, batch, _, _ = self.pool2(x, edge_index, edge_weight, batch)

        x = F.relu(self.conv3(x, edge_index, edge_weight=edge_weight))
        x, edge_index, edge_weight, batch, _, _ = self.pool3(x, edge_index, edge_weight, batch)

        x = F.relu(self.conv4(x, edge_index, edge_weight=edge_weight))
        x, edge_index, edge_weight, batch, _, _ = self.pool4(x, edge_index, edge_weight, batch)

        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        return self.linear(x)


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim1=32, hidden_dim2=15):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim1, bias=True)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2, bias=True)
        self.batch_norm1 = torch.nn.LayerNorm(hidden_dim1)
        self.batch_norm2 = torch.nn.LayerNorm(hidden_dim2)
        self.linear = torch.nn.Linear(hidden_dim2, 1)

        torch.nn.init.kaiming_uniform_(self.conv1.lin.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv2.lin.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x, edge_index, batch):
        x = F.leaky_relu(self.batch_norm1(self.conv1(x, edge_index)), negative_slope=0.01)
        x = F.leaky_relu(self.batch_norm2(self.conv2(x, edge_index)), negative_slope=0.01)
        x = global_mean_pool(x, batch)
        return self.linear(x)


class GCN_bare(torch.nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 1, bias=False)
        self.post_pool = torch.nn.Linear(1, 1, bias=True)

        torch.nn.init.xavier_uniform_(self.conv1.lin.weight)
        torch.nn.init.xavier_uniform_(self.post_pool.weight)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = global_add_pool(x, batch)
        return self.post_pool(x)


class GCN_deeper(torch.nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.conv2 = GCNConv(64, 64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.linear = torch.nn.Linear(64, 1)

        # Initialize weights (uniform)
        torch.nn.init.kaiming_uniform_(self.conv1.lin.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv2.lin.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='linear')
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.dropout(x)

        x = global_mean_pool(x, batch) 

        return self.linear(x) 




def debug_check(batch):
    if torch.isnan(batch.x).any():
        print("NaN detected in X")
    if torch.isinf(batch.x).any():
        print("Inf detected in X")
    if torch.isnan(batch.y).any():
        print("NaN detected in Y")
    if torch.isinf(batch.y).any():
        print("Inf detected in Y")
    if batch.edge_index.max() >= batch.x.shape[0]:
        print("Edge index out of bounds:", batch.edge_index.max().item(), "vs", batch.x.shape[0])



def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        debug_check(batch)


        out = model(batch.x, batch.edge_index, batch.batch)

        target = batch.y.view(-1, 1).float()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)



@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)  
        out = model(batch.x, batch.edge_index, batch.batch)
        
        target = batch.y.view(-1, 1).float()  
        loss = criterion(out, target)
        total_loss += loss.item()
    return total_loss / len(loader)


def train_model(model, train_loader, test_loader, epochs=100, patience=999, lr=0.001, weight_decay=0.0001, log=None, device='cpu'):
    criterion = torch.nn.MSELoss()
    #criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)

        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} {time.ctime(time.time())}", file=log, flush=True)


        if test_loss < best_loss:
            best_loss = test_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1


        if (patience != 999) & (patience_counter >= patience):
            print(f"Early stopping triggered after {epoch} epochs (best test loss: {best_loss:.6f})", file=log, flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print("Restored best model with test loss:", best_loss, file=log, flush=True)


    return model

def load_data(args):
    # ignore non args.use_rois option for now
    num_node_features = 1
    if args.partial_dat:
        if args.use_rois:
            bm_test = pd.read_csv("data_out/brain_mapping_mean_partial_test.csv")
            bm_train_all = pd.read_csv("data_out/brain_mapping_mean_partial_train.csv")
            bm_train, bm_val = train_test_split(bm_train_all, test_size=0.2, random_state=42)
            x_cols = [col for col in bm_train.columns if "label_" in col]

        else:
            FVE_df_org = pd.read_csv("data/FVE_dat_partial.csv")
            bm_train_all = FVE_df_org   # quick fix for args.use_rois == False
            bm_test = FVE_df_org
            x_cols = [col for col in FVE_df_org.columns if "l_" or "r_" in col]

        SurfeView_surfaces = scipy.io.loadmat("data/SurfeView_surfaces.mat")
        non_surface_area_vars = ["nihtbx_cryst_uncorrected"]


    elif args.partial_tsa_dat:
        if args.use_rois:
            bm_test = pd.read_csv("data_out/brain_mapping_mean_partial_tsa_test.csv")
            bm_train_all = pd.read_csv("data_out/brain_mapping_mean_partial_tsa_train.csv")
            bm_train, bm_val = train_test_split(bm_train_all, test_size=0.2, random_state=42)
            x_cols = [col for col in bm_train.columns if "label_" in col]

        else:
            FVE_df_org = pd.read_csv("data/FVE_dat_partial_tsa.csv")
            x_cols = [col for col in FVE_df_org.columns if "r_" or "l_" in col]

        SurfeView_surfaces = scipy.io.loadmat("data/SurfeView_surfaces.mat")
        non_surface_area_vars = ["nihtbx_cryst_uncorrected"]

    else:
        num_node_features = 3
        if args.use_rois:
            bm_test = pd.read_csv("data_out/brain_mapping_mean_regular_test.csv")
            bm_train_all = pd.read_csv("data_out/brain_mapping_mean_regular_train.csv")
            bm_train, bm_val = train_test_split(bm_train_all, test_size=0.2, random_state=42)
            x_cols = [col for col in bm_train.columns if "label_" in col]
            x_cols.append("interview_age")
            x_cols.append("sex_2")

        else:
            FVE_df_org = pd.read_csv("data/FVE_dat.csv")
            bm_train_all = FVE_df_org # quick fix for args.use_rois == False
            bm_test = FVE_df_org
            x_cols = [col for col in FVE_df_org.columns if "l_" or "r_" in col]
            x_cols.append("interview_age")
            x_cols.append("sex_2")      

        SurfeView_surfaces = scipy.io.loadmat("data/SurfeView_surfaces.mat")
        non_surface_area_vars = ["interview_age", "sex_2", "nihtbx_cryst_uncorrected"] 


    return bm_train_all, bm_test, SurfeView_surfaces, non_surface_area_vars, num_node_features, x_cols


def split_data(args, bm_train_all, bm_test, x_cols, i):
    if args.m == 1:
        bm_train, bm_val = train_test_split(bm_train_all, test_size=0.2, random_state=42)
        X_train = bm_train[x_cols]
        y_train = bm_train["nihtbx_cryst_uncorrected"]
        X_val = bm_val[x_cols]
        y_val = bm_val["nihtbx_cryst_uncorrected"]
        X_test = bm_test[x_cols]
        y_test = bm_test["nihtbx_cryst_uncorrected"]

    else:
        if args.use_rois:
            FVE_df_org = pd.concat([bm_train_all, bm_test])
        else:
            FVE_df_org = bm_train_all

        n = len(FVE_df_org)
        iter_rng = np.random.RandomState(args.seed + i) if args.seed is not None else np.random
        boot_indices = iter_rng.choice(n, size=n, replace=True)
        oob_indices = np.setdiff1d(np.arange(n), boot_indices)

        FVE_df_boot = FVE_df_org.iloc[boot_indices]
        FVE_df_train, FVE_df_val = train_test_split(FVE_df_boot, test_size=0.2, random_state=42)
        FVE_df_oob = FVE_df_org.iloc[oob_indices]

        X_train = FVE_df_train[x_cols]
        y_train = FVE_df_train["nihtbx_cryst_uncorrected"]
        X_val = FVE_df_val[x_cols]
        y_val = FVE_df_val["nihtbx_cryst_uncorrected"]
        X_test = FVE_df_oob[x_cols]
        y_test = FVE_df_oob["nihtbx_cryst_uncorrected"]

    print(f"input sanity check: partial={args.partial_dat}, partial_tsa={args.partial_tsa_dat}, x_cols={X_train.columns[0:1]} to {X_train.columns[(len(X_train.columns)-3):len(X_train.columns)]}")

    # standardization
    if args.x_scaler_name=="minmax":
        scaler_X = MinMaxScaler().fit(X_train)
        X_train_scaled = pd.DataFrame(scaler_X.transform(X_train), columns=x_cols, index=X_train.index)
        X_val_scaled   = pd.DataFrame(scaler_X.transform(X_val),   columns=x_cols, index=X_val.index)
        X_test_scaled  = pd.DataFrame(scaler_X.transform(X_test),  columns=x_cols, index=X_test.index)
    elif args.x_scaler_name=="standard":
        scaler_X = StandardScaler().fit(X_train)
        X_train_scaled = pd.DataFrame(scaler_X.transform(X_train), columns=x_cols, index=X_train.index)
        X_val_scaled   = pd.DataFrame(scaler_X.transform(X_val),   columns=x_cols, index=X_val.index)
        X_test_scaled  = pd.DataFrame(scaler_X.transform(X_test),  columns=x_cols, index=X_test.index)
    else:
        X_train_scaled = pd.DataFrame(X_train)
        X_test_scaled  = pd.DataFrame(X_test)
        X_val_scaled  = pd.DataFrame(X_val)

    # standardize y
    if args.y_scaler_name=="minmax":
        scaler_y = MinMaxScaler().fit(y_train.values.reshape(-1,1))
        y_train_scaled = scaler_y.transform(y_train.values.reshape(-1,1)).flatten()
        y_val_scaled   = scaler_y.transform(y_val.values.reshape(-1,1)).flatten()
        y_test_scaled  = scaler_y.transform(y_test.values.reshape(-1,1)).flatten()
    elif args.y_scaler_name=="standard":
        scaler_y = StandardScaler().fit(y_train.values.reshape(-1,1))
        y_train_scaled = scaler_y.transform(y_train.values.reshape(-1,1)).flatten()
        y_val_scaled   = scaler_y.transform(y_val.values.reshape(-1,1)).flatten()
        y_test_scaled  = scaler_y.transform(y_test.values.reshape(-1,1)).flatten()
    else:
        y_train_scaled, y_test_scaled, y_val_scaled = y_train, y_test, y_val
    
    y = [y_train_scaled, y_val_scaled, y_test_scaled]
    x = [X_train_scaled, X_val_scaled, X_test_scaled]

    return x, y


def main(args):
    if args.partial_dat:
        dat_type = "GCN_partial"
    elif args.partial_tsa_dat:
        dat_type = "GCN_partial_tsa"
    else:
        dat_type = "GCN_regular"


    job_full_nickname = f"{args.job_nickname}_{dat_type}_{args.model_type}_lr{args.lr}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    with open(f"GCN_models/{job_full_nickname}.txt", "w") as f:
        print(args, file=f, flush=True)
        print(f"Writing model output... using device: {device}", file=f, flush=True)
        print("{}".format(args).replace(', ', ',\n'), file=f, flush=True)

        # data load
        print(f"Loading data at {time.ctime(time.time())}", file=f, flush=True)
        bm_train_all, bm_test, SurfeView_surfaces, non_surface_area_vars, num_node_features, x_cols = load_data(args)

        R2_list = []
        MSE_list = []

        for i in range(args.m):
            print(f"-----------------------------STARTING bootstrap {i}th iteration -----------------------------", file=f, flush=True)
            x, y = split_data(args, bm_train_all, bm_test, x_cols, i)
            y_train_scaled, y_val_scaled, y_test_scaled = y
            X_train_scaled, X_val_scaled, X_test_scaled = x

            train_graph = input_to_graph(
                use_rois = args.use_rois,
                SurfeView_surfaces=SurfeView_surfaces,
                X_df_all=X_train_scaled,
                y_all = y_train_scaled,
                partial_dat=args.partial_dat,
                partial_tsa_dat=args.partial_tsa_dat,
                fs_mapping = args.fs_mapping,
                non_surface_area_vars = non_surface_area_vars
            )
            test_graph = input_to_graph(
                use_rois = args.use_rois,
                SurfeView_surfaces=SurfeView_surfaces,
                X_df_all=X_test_scaled,
                y_all = y_test_scaled,
                partial_dat=args.partial_dat,
                partial_tsa_dat=args.partial_tsa_dat,
                fs_mapping = args.fs_mapping,
                non_surface_area_vars = non_surface_area_vars
            )
            val_graph = input_to_graph(
                use_rois = args.use_rois,
                SurfeView_surfaces=SurfeView_surfaces,
                X_df_all=X_val_scaled,
                y_all = y_val_scaled,
                partial_dat=args.partial_dat,
                partial_tsa_dat=args.partial_tsa_dat,
                fs_mapping = args.fs_mapping,
                non_surface_area_vars = non_surface_area_vars
            )
            # Loaders
            train_loader = DataLoader(train_graph, batch_size=args.batch_size, shuffle=True, drop_last=True)
            val_loader   = DataLoader(val_graph, batch_size=args.batch_size, shuffle=False, drop_last=False)
            test_loader  = DataLoader(test_graph, batch_size=args.batch_size, shuffle=False, drop_last=False)

            print(f"Data loaded at {time.ctime(time.time())}", file=f, flush=True)


            print(f"num_node_features={num_node_features}")
            if args.model_type == "base":
                model = GCN_new(num_node_features=num_node_features, pool_ratio=args.pool_ratio).to(device)
            elif args.model_type == "base2":
                model = GCN_new2(num_node_features=num_node_features).to(device)
            elif args.model_type == "deep":
                model = GCN_deeper(num_node_features=num_node_features).to(device)
            elif args.model_type == "BrainGNN":
                model = Network(indim=num_node_features, ratio=0.8).to(device)

            # training
            print(f"Start training at {time.ctime(time.time())}", file=f, flush=True)


            trained_model = train_model(
                model=model,
                train_loader=train_loader,
                test_loader=val_loader,
                lr=args.lr,
                weight_decay=args.weight_decay,
                epochs=args.epochs,
                patience=args.patience,
                log=f,
                device=device,
            )

            # model save
            if args.m == 1:
                torch.save(trained_model, f"GCN_models/{job_full_nickname}.pt")


            print(f"Start eval at {time.ctime(time.time())}", file=f, flush=True)

            # eval
            trained_model.eval()
            y_pred, y_true = [], []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    out = trained_model(batch.x, batch.edge_index, batch.batch)
                    y_pred.extend(out.cpu().numpy().flatten())
                    y_true.extend(batch.y.cpu().numpy().flatten())

            y_pred, y_true = np.array(y_pred), np.array(y_true)
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            R2_list.append(r2)
            MSE_list.append(mse)
            
            print(f"-----------------------------AT M={i}: R2 Score: {r2:.4f}, MSE: {mse:.4f}-----------------------------", file=f, flush=True)
            print(f"-----------------------------AT M={i}, AVG R2 Score: {np.mean(R2_list):.4f} +/- {np.std(R2_list):.4f}-----------------------------", file=f, flush=True)

    
    if args.m > 1:
        pd.Series(R2_list).to_csv(f"/niddk-data-central/mae_hr/FVE/GCN_models/GCN_output/{job_full_nickname}_R2.csv")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_rois", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--pool_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1004)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--job_nickname", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="base",
                        choices=["base", "deep", "BrainGNN", "new", "base2"])
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--partial_dat", action="store_true")
    parser.add_argument("--partial_tsa_dat", action="store_true")
    parser.add_argument("--x_scaler_name", type=str, default="standard",
                        choices=["standard", "minmax", "none"],)
    parser.add_argument("--y_scaler_name", type=str, default="standard",
                        choices=["standard", "minmax", "none"],)
    parser.add_argument("--fs_mapping", action="store_true", default=False)
    parser.add_argument("--bootstrap", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
