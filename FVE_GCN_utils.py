import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
import numpy as np
import pandas as pd
import sys
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.optim as optim
import glob
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import time
from torch_geometric.nn import NNConv as MyNNConv, TopKPooling
import scipy.sparse as sp
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import scipy.io
from nilearn import datasets, surface
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)


def load_surface_mesh(SurfeView_surfaces):
    # coords
    coords_lh = SurfeView_surfaces['surf_lh_pial'][0,0]['vertices'][:10242]
    coords_rh = SurfeView_surfaces['surf_rh_pial'][0,0]['vertices'][:10242]

    # faces
    faces_lh = SurfeView_surfaces['icsurfs'][0,5]['faces'][0,0] - 1
    faces_rh = SurfeView_surfaces['icsurfs'][0,5]['faces'][0,0] - 1 + 10242

    # concatenate for left and right
    coords = np.concatenate([coords_lh, coords_rh])
    faces  = np.concatenate([faces_lh, faces_rh])
    return coords, faces




def build_surface_adjacency(coords, faces):
    """
    Build an adjacency list (or adjacency matrix) from surface mesh faces.
    
    Args:
        coords (ndarray): (n_vertices, 3) array of vertex coordinates
        faces (ndarray): (n_faces, 3) array, each row has 3 vertex indices forming a triangular face
    
    Returns:
        adjacency_matrix (ndaray): sparse adjacency_matrix is a set of vertices adjacent to vertex v
    """
    n_vertices = coords.shape[0]
    adjacency = [set() for _ in range(n_vertices)]

    # For each face, connect its three vertices (v1, v2, v3)
    for (v1, v2, v3) in faces:
        adjacency[v1].add(v2)
        adjacency[v1].add(v3)
        adjacency[v2].add(v1)
        adjacency[v2].add(v3)
        adjacency[v3].add(v1)
        adjacency[v3].add(v2)

    n_vertices = len(adjacency)
    rows = []
    cols = []
    for v in range(n_vertices):
        for w in adjacency[v]:
            rows.append(v)
            cols.append(w)
    data = np.ones(len(rows), dtype=int)
    
    # Build a square adjacency matrix
    A = sp.csr_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))
    
    # Symmetrize (sometimes needed if faces produce directed edges—but typically it's already symmetric)
    # A = A.maximum(A.T)
    
    return A


def map_to_fs(FVE_df_old_X, coords_fs_left, coords_fs_right, vertices_coords_orgs_left, vertices_coords_orgs_right, non_surface_area_vars):
    # calculate the mapping
    dist_matrix_left = distance.cdist(vertices_coords_orgs_left, coords_fs_left, 'euclidean')
    dist_matrix_right = distance.cdist(vertices_coords_orgs_right, coords_fs_right, 'euclidean')

    # mapping from original -> freesurfer (col is the assignment to free surfer index)
    row_ind_left, col_ind_left = linear_sum_assignment(dist_matrix_left)
    row_ind_right, col_ind_right = linear_sum_assignment(dist_matrix_right)

    ## create new df conforming to fs
    left_cols = [c for c in FVE_df_old_X.columns if c.endswith('_l')]
    right_cols = [c for c in FVE_df_old_X.columns if c.endswith('_r')]
    left_cols = sorted(left_cols, key=lambda x: int(x.split('_')[0]))
    right_cols = sorted(right_cols, key=lambda x: int(x.split('_')[0]))

    # Extract data as numpy arrays
    FVE_df_fs_X = pd.DataFrame(index=FVE_df_old_X.index)

    for i, new_idx in enumerate(col_ind_left):
        old_col = f"{i}_l"          
        new_col = f"{new_idx}_l"    
        FVE_df_fs_X[new_col] = FVE_df_old_X[old_col]

    for i, new_idx in enumerate(col_ind_right):
        old_col = f"{i}_r"
        new_col = f"{new_idx}_r"
        FVE_df_fs_X[new_col] = FVE_df_old_X[old_col]


    FVE_df_fs_X = FVE_df_fs_X.reindex(sorted(FVE_df_fs_X.columns, key=lambda x: (x.split("_")[1], int(x.split("_")[0]))), axis=1)
    FVE_df = pd.concat([FVE_df_fs_X, FVE_df_old_X[non_surface_area_vars]], axis=1)

    return FVE_df


def input_to_graph(SurfeView_surfaces, FVE_df_all, non_surface_area_vars=None, norm_y=False, scaler="minmax", fs_mapping=False,
                   partial_dat=False, partial_tsa_dat=False, icld_age_sex=False, batch_size=20):


    if fs_mapping:
        # original mapping
        vertices_coords_orgs, faces_orgs = load_surface_mesh(SurfeView_surfaces)
        vertices_coords_orgs_left = vertices_coords_orgs[:10242]
        vertices_coords_orgs_right = vertices_coords_orgs[10242:]

        # FreeSurfer
        fsaverage = datasets.fetch_surf_fsaverage('fsaverage5')
        coords_fs_left, faces_fs_left = surface.load_surf_mesh(fsaverage.pial_left)
        coords_fs_right, faces_fs_right = surface.load_surf_mesh(fsaverage.pial_right)
        vertices_coords = np.vstack([coords_fs_left, coords_fs_right])
        faces = np.vstack([faces_fs_left, faces_fs_right+10242])


        # graph structure
        A_csr = build_surface_adjacency(vertices_coords, faces)
        edge_index = torch.tensor(np.array(A_csr.nonzero()), dtype=torch.long)

    else:
        # original mapping
        vertices_coords, faces = load_surface_mesh(SurfeView_surfaces)
        # graph structure
        A_csr = build_surface_adjacency(vertices_coords, faces)
        edge_index = torch.tensor(np.array(A_csr.nonzero()), dtype=torch.long)


    # define x columns
    x_cols = [col for col in FVE_df.columns if ("_r" in col) or ("_l" in col)]

    if (not partial_dat and not partial_tsa_dat):
        x_cols.append("interview_age")
        x_cols.append("sex_2")


    if fs_mapping:
        FVE_df_old_X = FVE_df_all.dropna()
        FVE_df = map_to_fs(FVE_df_old_X, coords_fs_left, coords_fs_right, vertices_coords_orgs_left, vertices_coords_orgs_right, non_surface_area_vars)
    else:
        FVE_df = FVE_df_all.dropna()


    # train, val, test split
    train_valid, test = train_test_split(FVE_df, test_size=0.2, random_state=42)
    train, val = train_test_split(train_valid, test_size=0.2, random_state=42)
    
    

    # Normalization
    X_train = train[x_cols]
    X_val   = val[x_cols]
    X_test  = test[x_cols]

    y_train = train["nihtbx_cryst_uncorrected"]
    y_val   = val["nihtbx_cryst_uncorrected"]
    y_test  = test["nihtbx_cryst_uncorrected"]


    # standardization
    if scaler=="minmax":
        scaler_X = MinMaxScaler().fit(X_train)
    else:
        scaler_X = StandardScaler().fit(X_train)


    X_train_scaled = pd.DataFrame(scaler_X.transform(X_train), columns=x_cols, index=X_train.index)
    X_val_scaled   = pd.DataFrame(scaler_X.transform(X_val),   columns=x_cols, index=X_val.index)
    X_test_scaled  = pd.DataFrame(scaler_X.transform(X_test),  columns=x_cols, index=X_test.index)


    # standardize y
    if norm_y:
        if scaler=="minmax":
            scaler_y = MinMaxScaler().fit(y_train.values.reshape(-1,1))
        else:
            scaler_y = StandardScaler().fit(y_train.values.reshape(-1,1))
        y_train_scaled = scaler_y.transform(y_train.values.reshape(-1,1)).flatten()
        y_val_scaled   = scaler_y.transform(y_val.values.reshape(-1,1)).flatten()
        y_test_scaled  = scaler_y.transform(y_test.values.reshape(-1,1)).flatten()
    else:
        y_train_scaled, y_val_scaled, y_test_scaled = y_train, y_val, y_test

    print(f"sanity check before create_graph_data: y and x shape: {y_train_scaled.shape}, {X_train_scaled.shape}; x_cols: {X_train_scaled.columns[0:1]} to {X_train_scaled.columns[20483:]}")
    
    # Graph
    train_graph = create_graph_data(X_all=X_train_scaled, y=torch.Tensor(y_train_scaled), partial_dat=partial_dat, partial_tsa_dat= partial_tsa_dat,
                            edge_index = edge_index, vertices = torch.FloatTensor(vertices_coords), icld_age_sex=icld_age_sex)
    validation_graph = create_graph_data(X_all=X_val_scaled, y=torch.Tensor(y_val_scaled), partial_dat=partial_dat, partial_tsa_dat= partial_tsa_dat,
                                    edge_index = edge_index, vertices = torch.FloatTensor(vertices_coords), icld_age_sex=icld_age_sex)
    test_graph = create_graph_data(X_all=X_test_scaled, y=torch.Tensor(y_test_scaled), partial_dat=partial_dat, partial_tsa_dat= partial_tsa_dat,
                                    edge_index = edge_index, vertices = torch.FloatTensor(vertices_coords), icld_age_sex=icld_age_sex)    


    # Loaders
    train_loader = DataLoader(train_graph, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(validation_graph, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_graph, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader


def create_graph_data(X_all, y, edge_index, vertices, partial_dat=False, partial_tsa_dat=False, icld_age_sex=False):
    graphs = []
    
    #pos
    pos = vertices
    
    #edge_attr
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1, keepdim=True)
    
    #node features: X, interview_age, and sex
    if (partial_dat or partial_tsa_dat):
        X = torch.tensor(X_all.to_numpy(), dtype=torch.float32).unsqueeze(-1)
    else:
        X = torch.tensor(X_all.drop(['interview_age', 'sex_2'], axis=1).to_numpy(), dtype=torch.float32).unsqueeze(-1)

    if icld_age_sex:
        interview_age = torch.tensor(X_all["interview_age"].to_numpy())
        sex = torch.tensor(X_all["sex_2"].to_numpy())
        age_expanded = interview_age.unsqueeze(1).repeat(1, X.shape[1]).unsqueeze(2)
        sex_expanded = sex.unsqueeze(1).repeat(1, X.shape[1]).unsqueeze(2)           
        node_feat = torch.tensor(torch.cat([X, age_expanded, sex_expanded], dim=2), dtype=torch.float32)
    else:
        node_feat = torch.tensor(X, dtype=torch.float32)

    print(f"creat_graph_data sanity check: icld_age_sex = {icld_age_sex}, node_feat.shape: {node_feat.shape}")
    for i in range(node_feat.shape[0]):        
        data = Data(
            x=node_feat[i],
            edge_index=edge_index,
            pos = pos,
            #edge_attr = edge_attr,
            y= torch.FloatTensor(y[i])
        )
        graphs.append(data)
    return graphs



def summarize_by_brain_region(df, data_l, data_r):

    df_l = df[[f"{i}_l" for i in range(len(data_l))]].copy()
    df_r = df[[f"{i}_r" for i in range(len(data_r))]].copy()

    X_l = df_l.to_numpy()  
    X_r = df_r.to_numpy()  

    # All labels
    unique_labels = np.unique(np.concatenate([data_l, data_r]))

    result = {}

    for label in unique_labels:

        # indices in each hemisphere
        idx_l = np.where(data_l == label)[0]
        idx_r = np.where(data_r == label)[0]

        # gather data for all vertices with this label
        verts = []
        if len(idx_l) > 0:
            verts.append(X_l[:, idx_l])
        if len(idx_r) > 0:
            verts.append(X_r[:, idx_r])

        # concatenate across vertices (second axis)
        combined = np.concatenate(verts, axis=1)  # shape: (N, n_vertices_for_label)

        # average across vertices
        result[f"label_{label}"] = combined.mean(axis=1)

    # final summarized df
    return pd.DataFrame(result, index=df.index)