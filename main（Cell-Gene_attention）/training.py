import sys, os

import numpy as np
 
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import HeteroData
from torch_geometric.data import Data
import os
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc, f1_score, precision_score, recall_score
from sklearn.preprocessing import normalize
from scipy.linalg import block_diag
import numpy as np
import wandb
from tqdm import trange
import torch_geometric.transforms as T
import pandas as pd

        
####################################################
def create_pyg_data(preprocessing_output_folderpath, split=0.1):

    if not os.path.exists(preprocessing_output_folderpath) or \
        not {"celllevel_adjacencylist.npy","celllevel_adjacencymatrix1.npy",  "celllevel_edgelist1.npy", "genelevel_edgelist.npy", "celllevel_features.npy", "genelevel_features.npy"}.issubset(set(os.listdir(preprocessing_output_folderpath))):
        
        raise Exception("Proper preprocessing files not found. Please run the 'preprocessing' step.")
    
    # Load preprocessing files
    celllevel_adjacencymatrix = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath, "celllevel_adjacencymatrix1.npy"))).type(torch.LongTensor)
    celllevel_features = torch.from_numpy(normalize(np.load(os.path.join(preprocessing_output_folderpath, "celllevel_features.npy")))).type(torch.float32)
    celllevel_edgelist = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath, "celllevel_edgelist1.npy"))).type(torch.LongTensor)
    genelevel_edgelist = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath, "genelevel_edgelist.npy"))).type(torch.LongTensor)
    genelevel_features = torch.from_numpy(normalize(np.load(os.path.join(preprocessing_output_folderpath, "genelevel_features.npy")))).type(torch.float32)
    genelevel_grns_flat = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath, "initial_grns.npy"))).type(torch.float32).flatten()
    celllevel_edgelist_1 = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath, "celllevel_edgelist2.npy"))).type(torch.LongTensor)
    celllevel_adjacencymatrix_1 = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath, "celllevel_adjacencymatrix2.npy"))).type(torch.LongTensor)
    celllevel_edgelist_2 = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath, "celllevel_edgelist3.npy"))).type(torch.LongTensor)
    celllevel_adjacencymatrix_2 = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath, "celllevel_adjacencymatrix3.npy"))).type(torch.LongTensor) 
    celllevel_edgelist_3 = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath, "celllevel_edgelist4.npy"))).type(torch.LongTensor)
    celllevel_adjacencymatrix_3 = torch.from_numpy(np.load(os.path.join(preprocessing_output_folderpath, "celllevel_adjacencymatrix4.npy"))).type(torch.LongTensor) 

    print("celllevel_adjacencymatrix shape:", celllevel_adjacencymatrix.shape)
    print("celllevel_features shape:", celllevel_features.shape)
    print("celllevel_edgelist shape:", celllevel_edgelist.shape)
    print("genelevel_edgelist shape:", genelevel_edgelist.shape)
    print("genelevel_features shape:", genelevel_features.shape)
    print("genelevel_grns_flat shape:", genelevel_grns_flat.shape)
    print("celllevel_edgelist_1 shape:", celllevel_edgelist_1.shape)
    print("celllevel_adjacencymatrix_1 shape:", celllevel_adjacencymatrix_1.shape)
    print("celllevel_edgelist_2 shape:", celllevel_edgelist_2.shape)
    print("celllevel_adjacencymatrix_2 shape:", celllevel_adjacencymatrix_2.shape)
    print("celllevel_edgelist_3 shape:", celllevel_edgelist_3.shape)
    print("celllevel_adjacencymatrix_3 shape:", celllevel_adjacencymatrix_3.shape)


    cell_level_data = Data(x=celllevel_features, edge_index=celllevel_edgelist, y=celllevel_adjacencymatrix)
    gene_level_data = Data(x=genelevel_features, edge_index=genelevel_edgelist, y=genelevel_grns_flat)
    cell_level_data1 = Data(x=celllevel_features, edge_index=celllevel_edgelist_1, y=celllevel_adjacencymatrix_1)
    cell_level_data2 = Data(x=celllevel_features, edge_index=celllevel_edgelist_2, y=celllevel_adjacencymatrix_2)
    cell_level_data3 = Data(x=celllevel_features, edge_index=celllevel_edgelist_3, y=celllevel_adjacencymatrix_3)

    if split is not None:
        print(f"{1-split} training edges | {split} testing edges")
        transform = T.RandomLinkSplit(
            num_test=split,
            num_val=0,
            is_undirected=True, 
            add_negative_train_samples=True,
            neg_sampling_ratio=1,
            key="edge_label",  # supervision label
            disjoint_train_ratio=0.7,
        )



        train_cell_level_data3, _, test_cell_level_data3 = transform(cell_level_data3)
        cell_level_data3 = (train_cell_level_data3, test_cell_level_data3)

    return cell_level_data, cell_level_data1, cell_level_data2,cell_level_data3, gene_level_data


def train_gae(data, model, hyperparameters):
        # wandb.init()
        # wandb.config = hyperparameters
        num_epochs = hyperparameters["num_epochs"]
        optimizer = hyperparameters["optimizer"]
        criterion = hyperparameters["criterion"]
        split = hyperparameters["split"]
        num_genespercell = hyperparameters["num_genespercell"]
        device = hyperparameters.get("device", "cpu")

        if split is not None:
            cell_level_data = data[0].to(device)
            cell_level_data1 = data[1].to(device)
            cell_level_data2 = data[2].to(device)
            cell_train_data3 = data[3][0].to(device)
            cell_test_data3 = data[3][1].to(device)
        else:
            cell_train_data = data[0].to(device)
        gene_train_data = data[4].to(device)
        
        num_cells = cell_train_data3.x.shape[0]
        intracellular_gene_mask = torch.from_numpy(create_intracellular_gene_mask(num_cells, num_genespercell)).to(device)
        # Get the indices for the intracellular edges for the loss calculation
        intracellular_edge_index = torch.from_numpy(np.array(np.where(intracellular_gene_mask.cpu().numpy()))).to(device)
        intracellular_gene_mask = intracellular_gene_mask.to(device)

        mse = torch.nn.MSELoss()
        test_roc_scores = []
        test_ap_scores = []
        test_auprc_scores = []
        test_f1_scores = []
        test_precision_scores = []
        test_recall_scores = []

        with trange(num_epochs,desc="") as pbar:
            for epoch in pbar:
                pbar.set_description(f"Epoch {epoch}")
                model.train()
                optimizer.zero_grad()  # Clear gradients.
                posmask = cell_train_data3.edge_label == 1
                # posmask1 = cell_train_data.edge_label == 1
                # posmask2 = cell_train_data1.edge_label == 1
                # posmask3 = cell_train_data2.edge_label == 1
                # Unpack model outputs (a is alignment_loss, c is contrast_loss)
                z, _, _, z_g, a, c = model.encode(cell_train_data3.x, gene_train_data.x, cell_level_data.edge_index, gene_train_data.edge_index,cell_level_data1.edge_index,cell_level_data2.edge_index,gene_train_data.edge_index)

                # Build Loss - reconstruction + auxiliary losses
                # 1. Start with reconstruction loss
                recon_loss = model.recon_loss(z, cell_train_data3.edge_label_index[:, posmask])
                loss = recon_loss

                # 2. Add intracellular gene reconstruction loss
                recon_Ag_intra = model.decode(z_g, intracellular_edge_index)
                true_Ag_intra = gene_train_data.y
                intracellular_penalty_loss = mse(recon_Ag_intra, true_Ag_intra)
                loss = loss + intracellular_penalty_loss

                # 3. Add auxiliary losses with fixed weights
                loss = loss + 0.1 * a + 0.1 * c
                
                auroc,ap = model.test(z,  cell_train_data3.edge_label_index[:, posmask], cell_train_data3.edge_label_index[:,~posmask])

              
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.


                # Test every epoch
                model.eval()
                posmask = cell_test_data3.edge_label == 1
                test_recon_loss = model.recon_loss(z, cell_test_data3.edge_label_index[:, posmask])   # negative edges automatically sampled in 1:1 correspondence with positive edges according to pyg documentation
                test_rocauc, test_ap = model.test(z, cell_test_data3.edge_label_index[:, posmask], cell_test_data3.edge_label_index[:,~posmask])
                test_precision, test_recall, _ = precision_recall(model,z, cell_test_data3.edge_label_index[:, posmask], cell_test_data3.edge_label_index[:,~posmask])
                test_auprc = auc(test_recall, test_precision)

                # Calculate F1, Precision, Recall at 0.5 threshold
                pos_y = z.new_ones(cell_test_data3.edge_label_index[:, posmask].size(1))
                neg_y = z.new_zeros(cell_test_data3.edge_label_index[:, ~posmask].size(1))
                y_true = torch.cat([pos_y, neg_y], dim=0).cpu().numpy()

                pos_pred = model.decode(z, cell_test_data3.edge_label_index[:, posmask], sigmoid=True)
                neg_pred = model.decode(z, cell_test_data3.edge_label_index[:, ~posmask], sigmoid=True)
                y_pred_proba = torch.cat([pos_pred, neg_pred], dim=0).detach().cpu().numpy()
                y_pred_binary = (y_pred_proba >= 0.5).astype(int)

                test_f1 = f1_score(y_true, y_pred_binary, zero_division=0)
                test_prec = precision_score(y_true, y_pred_binary, zero_division=0)
                test_rec = recall_score(y_true, y_pred_binary, zero_division=0)

                test_roc_scores.append(test_rocauc)
                test_auprc_scores.append(test_auprc)
                test_ap_scores.append(test_ap)
                test_f1_scores.append(test_f1)
                test_precision_scores.append(test_prec)
                test_recall_scores.append(test_rec)

                pbar.set_postfix(train_loss=loss.item(), train_recon_loss = recon_loss.item(),test_recon_loss =test_recon_loss.item())

        metrics_df = pd.DataFrame({
            "Epoch": range(num_epochs),
            "Test AP": test_ap_scores,
            "Test ROC": test_roc_scores,
            "Test AUPRC": test_auprc_scores,
            "Test F1": test_f1_scores,
            "Test Precision": test_precision_scores,
            "Test Recall": test_recall_scores
        })
        
        return model, metrics_df


def precision_recall(model, z, pos_edge_index, neg_edge_index):
    pos_y = z.new_ones(pos_edge_index.size(1))
    neg_y = z.new_zeros(neg_edge_index.size(1))
    y = torch.cat([pos_y, neg_y], dim=0)

    pos_pred = model.decode(z, pos_edge_index, sigmoid=True)
    neg_pred = model.decode(z, neg_edge_index, sigmoid=True)
    pred = torch.cat([pos_pred, neg_pred], dim=0)

    y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

    return precision_recall_curve(y, pred)


      
def create_intracellular_gene_mask(num_cells, num_genespercell):
  I = np.ones(shape=(num_genespercell,num_genespercell))
  block_list = [I for _ in range(num_cells)]
  return block_diag(*block_list).astype(bool)