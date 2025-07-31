import numpy as np
import pandas as pd
import networkx as nx
import os
import argparse
from rich.table import Table
from rich.console import Console
from rich.text import Text
from rich.spinner import Spinner
import random
import torch
import matplotlib.pyplot as plt

import time
import preprocessing as preprocessing
import training as training
import models as models

from torch_geometric.nn import GAE

debug = False
def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # All GPUs (if applicable)
    torch.backends.cudnn.deterministic = True  # For deterministic results on GPU
    torch.backends.cudnn.benchmark = False  # Disables the inferences of the optimum algorithms for the hardware

# Set the seed
set_random_seed(1)  
def parse_arguments():
    parser = argparse.ArgumentParser(description='clarifyGAE arguments')
    parser.add_argument("-m", "--mode", type=str, default = "train",
        help="clarifyGAE mode: preprocess,train,test")
    parser.add_argument("-i", "--inputdirpath", type=str,
                   help="Input directory path where ST data is stored")
    parser.add_argument("-o", "--outputdirpath", type=str,
                   help="Output directory path where results will be stored ")
    parser.add_argument("-s", "--studyname", type=str,
                    help="clarifyGAE study name")
    parser.add_argument("-t", "--split", type=float,
        help="# of test edges [0,1)")
    parser.add_argument("-n", "--numgenespercell", type=int, default = 20,
               help="Number of genes in each gene regulatory network")
    parser.add_argument("-k", "--nearestneighbors", type=int, default = 2,
               help="Number of nearest neighbors for each cell")
    parser.add_argument("-l", "--lrdatabase", type=int, default=0,
                   help="0/1/2 for which Ligand-Receptor Database to use")
    parser.add_argument("-a", "--ownadjacencypath", type=str,
        help="Using your own cell level adjacency (give path)")
    parser.add_argument("--edge_threshold_validation", type=float, default=0.5,
                      help="edge threshold for validation [0,1)")
    parser.add_argument("--validation_cpu_cores", type=int, default=4,
                      help="number of CPU cores for validation")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for GCN layers")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for training (e.g., 'cpu', 'cuda', 'cuda:1')")
    args = parser.parse_args()
    return args



def preprocess(st_data, num_nearestneighbors, lrgene_ids, cespgrn_hyperparameters, ownadjacencypath = None):    
        
    # 1. Infer initial GRNs with CeSpGRN
    if debug: # skip inferring GRNs while debugging
        print("1. Skipping CeSpGRN inference (for debug mode)")
        grns = np.load("../out/scmultisim_final/1_preprocessing_output/initial_grns.npy")
    else:
        grns = preprocessing.infer_initial_grns(st_data, cespgrn_hyperparameters) # shape (ncells, ngenes, ngenes)

    # 2. Construct Cell-Level Graph from ST Data
    if ownadjacencypath is not None:
        celllevel_adj = np.load(ownadjacencypath)
    else:
        celllevel_adj, _ = preprocessing.construct_celllevel_graph(st_data, num_nearestneighbors, get_edges=False)

    #  3. Construct Gene-Level Graph from ST Data + GRNs 
    gene_level_graph, num2gene, gene2num, grn_components = preprocessing.construct_genelevel_graph(grns, celllevel_adj, node_type="int", lrgenes = lrgene_ids)
    
    # 4. Generate Gene Feature vectors
    gene_features, genefeaturemodel = preprocessing.get_gene_features(grn_components, type="node2vec")
    

    return celllevel_adj, gene_level_graph, num2gene, gene2num, grns, gene_features, genefeaturemodel




def build_clarifyGAE_pytorch(data, hyperparams = None):

    train_data = data[3][0] if isinstance(data[3], tuple) else data[3]
    num_cells1 = train_data.x.shape[0]
    num_cellfeatures1 = train_data.x.shape[1]
    
    gene_data = data[4]
    print(f"Number of cells detected: {num_cells1}, Number of features: {num_cellfeatures1}")
    print(f"Gene data y size: {gene_data.y.size()}")
    
    num_genes, num_genefeatures = data[4].x.shape[0], data[4].x.shape[1]
    hidden_dim = hyperparams["concat_hidden_dim"] 
    num_genespercell = hyperparams["num_genespercell"]
    hidden_dim1 = hyperparams["concat_hidden_dim"]
    dropout = hyperparams.get("dropout", 0.14)

    # Extract only the necessary hyperparameters for the model
    contrastive_temp = hyperparams.get("contrastive_temp", 0.2)
    contrastive_hard_weight = hyperparams.get("contrastive_hard_weight", 0.8)

    cellEncoder = models.GraphEncoder(num_cellfeatures1, hidden_dim, dropout=dropout)
    geneEncoder = models.SubgraphEncoder(num_features=num_genefeatures, hidden_dim=hidden_dim, num_vertices = num_cells1, num_subvertices = num_genespercell, dropout=dropout)
    
    # Build the encoder with simplified, reconstruction-focused parameters
    multiviewEncoder = models.MultiviewEncoder(
        SubgraphEncoder=geneEncoder,
        GraphEncoder=cellEncoder,
        dropout=dropout,
        contrastive_temp=contrastive_temp,
        contrastive_hard_weight=contrastive_hard_weight
    )
    gae = GAE(multiviewEncoder)

    return gae



def main():
    console = Console()
    text = Text.from_ansi(r"""
    --------------------------------------------------------------------------------
PLOSCB！！！！！！！！！！！！！！！
    --------------------------------------------------------------------------------
                    """)
    text.stylize("cyan")
    console.print(text)
    
    with console.status("[cyan]booting up ...") as status:
        status.update(spinner="aesthetic", spinner_style="cyan")
        time.sleep(4)
        status.update(status="[cyan] arguments ...")
        args = parse_arguments()
        
        # --- Device Setup ---
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        print(f"Using device: {device}")

        mode = args.mode
        input_dir_path = args.inputdirpath
        output_dir_path = args.outputdirpath
        num_nearestneighbors = args.nearestneighbors
        num_genespercell = args.numgenespercell
        LR_database = args.lrdatabase
        studyname = args.studyname
        ownadjacencypath = args.ownadjacencypath
        split = args.split
        
        preprocess_output_path = os.path.join(output_dir_path, "1_preprocessing_output")
        training_output_path = os.path.join(output_dir_path, "2_training_output")
        evaluation_output_path = os.path.join(output_dir_path, "3_evaluation_output")
        time.sleep(4)
        status.update(status="[cyan] Done")
        time.sleep(2)

        
    if "preprocess" in mode:
        if not os.path.exists(preprocess_output_path):
            os.mkdir(preprocess_output_path)
        start_time = time.time()
        print("\n#------------------------------------ Loading in data ---------------------------------#\n")
        st_data = pd.read_csv(input_dir_path, index_col=None)
        assert {"Cell_ID", "X", "Y", "Cell_Type"}.issubset(set(st_data.columns.to_list()))
        
        numcells, totalnumgenes = st_data.shape[0], st_data.shape[1] - 4
        print(f"{numcells} Cells & {totalnumgenes} Total Genes\n")
        
        cespgrn_hyperparameters = {
            "bandwidth" : 0.1,
            "n_neigh" : 30,
            "lamb" : 0.1,
            "max_iters" : 1000
        }
        
        print(f"Hyperparameters:\n # of Nearest Neighbors: {num_nearestneighbors}\n # of Genes per Cell: {num_genespercell}\n")
        selected_st_data, lrgene2id = preprocessing.select_LRgenes(st_data, num_genespercell, LR_database)

        print("\n#------------------------------------ Preprocessing ----------------------------------#\n")
        
        if not os.path.exists(preprocess_output_path):
            os.mkdir(preprocess_output_path)
        celllevel_coordinates = st_data[["Cell_ID","X", "Y"]].values
        np.save(os.path.join(preprocess_output_path, "celllevel_coordinates.npy"), celllevel_coordinates)
        celllevel_features = st_data.drop(["Cell_ID", "Cell_Type", "X", "Y"], axis = 1).values

        celllevel_adj, genelevel_graph, num2gene, gene2num, grns, genelevel_features, genelevel_feature_model = preprocess(selected_st_data, num_nearestneighbors,lrgene2id.values(), cespgrn_hyperparameters, ownadjacencypath)
            
        celllevel_edgelist = preprocessing.convert_adjacencylist2edgelist(celllevel_adj)
        genelevel_edgelist = nx.to_pandas_edgelist(genelevel_graph).drop(["weight"], axis=1).to_numpy().T
        genelevel_adjmatrix = nx.adjacency_matrix(genelevel_graph, weight=None)
        
        assert celllevel_edgelist.shape == (2, celllevel_adj.shape[0] * celllevel_adj.shape[1])
        
        np.save(os.path.join(preprocess_output_path, "celllevel_adjacencylist.npy"),celllevel_adj)
        np.save(os.path.join(preprocess_output_path, "celllevel_adjacencymatrix.npy"),preprocessing.convert_adjacencylist2adjacencymatrix(celllevel_adj))
        np.save(os.path.join(preprocess_output_path, "celllevel_edgelist.npy"),celllevel_edgelist)
        np.save(os.path.join(preprocess_output_path, "celllevel_features.npy"),celllevel_features)
        np.save(os.path.join(preprocess_output_path, "genelevel_edgelist.npy"),genelevel_edgelist)
        np.save(os.path.join(preprocess_output_path, "genelevel_adjmatrix.npy"),genelevel_adjmatrix)
        np.save(file = os.path.join(preprocess_output_path, "initial_grns.npy"), arr = grns) 
        
        np.save(os.path.join(preprocess_output_path, "genelevel_features.npy"), genelevel_features) 
        genelevel_feature_model.save(os.path.join(preprocess_output_path, "genelevel_feature_model")) 

        print(f"Finished preprocessing in {(time.time() - start_time)/60} mins.\n")
    

    if "train" in mode:
        # Initial hyperparameters
        hyperparameters = {
            "num_genespercell": num_genespercell,
            "concat_hidden_dim": 32,
            "dropout": args.dropout,
            "criterion": torch.nn.BCELoss(),
            "num_epochs": 120,
            "split": args.split,
            "device": device,
        }

        print("\n#------------------------------ Creating PyG Datasets ----------------------------#\n")

        celllevel_data, celllevel_data1, celllevel_data2, cell_level_data3, genelevel_data = training.create_pyg_data(
            preprocess_output_path, hyperparameters["split"]
        )
        
        console = Console()
        table = Table(show_header=True, header_style="bold")
        table.add_column("Cell Level PyG Data", style="cyan")
        table.add_column("Gene Level PyG Data", style="deep_pink3")
        
        celllevel_str = str(cell_level_data3)
        table.add_row(celllevel_str, "".join(str(genelevel_data).split("\n")))
        console.print(table)

        if not os.path.exists(training_output_path):
            os.mkdir(training_output_path)

        print("\n#------------------------------- Training -----------------------------#\n")

        data = (celllevel_data, celllevel_data1, celllevel_data2, cell_level_data3, genelevel_data)

        # 1. Build the model first
        model = build_clarifyGAE_pytorch(data, hyperparameters).to(device)
        print(model)

        # 2. Now that the model exists, create the optimizer and add it to hyperparameters
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0098)
        hyperparameters["optimizer"] = optimizer
        
        # 3. Start training
        trained_model, metrics_df = training.train_gae(
            model=model, data=data, hyperparameters=hyperparameters
        )
        
        torch.save(trained_model.state_dict(), os.path.join(training_output_path,f'{studyname}_trained_gae_model_{split}.pth'))

        if not os.path.exists(evaluation_output_path):
            os.mkdir(evaluation_output_path)
            
        metrics_df.to_csv(os.path.join(evaluation_output_path, f"{studyname}_metrics_{args.split}.csv"))

        generate_reconstructed_matrix(trained_model, data, training_output_path)

def generate_reconstructed_matrix(model, data, output_path):

    
    device = next(model.parameters()).device
    print(f": {device}")
    
    cell_level_data = data[0].to(device)
    cell_level_data1 = data[1].to(device)
    cell_level_data2 = data[2].to(device)
    
    cell_test_data3 = data[3][1] if isinstance(data[3], tuple) else data[3]
    cell_test_data3 = cell_test_data3.to(device)
    gene_train_data = data[4].to(device)
    
    cell_features = cell_test_data3.x
    gene_features = gene_train_data.x
    
    model.eval()
    
    with torch.no_grad():
        print("Generating node embeddings...")
        z, _, _, z_g, _, _ = model.encode(
            cell_features, 
            gene_features, 
            cell_level_data.edge_index,
            gene_train_data.edge_index,
            cell_level_data1.edge_index,
            cell_level_data2.edge_index,
            gene_train_data.edge_index
        )
        
        print("Generating reconstructed adjacency matrices...")
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'forward_all'):
            adj_reconstructed_tensor = model.decoder.forward_all(z)
            adj_reconstructed_tensor = torch.sigmoid(adj_reconstructed_tensor)
            adj_reconstructed_np = adj_reconstructed_tensor.cpu().numpy()
            print(f"Reconstructed adjacency matrix shape: {adj_reconstructed_np.shape}")
            
            gene_adj_reconstructed_tensor = model.decoder.forward_all(z_g)
            gene_adj_reconstructed_tensor = torch.sigmoid(gene_adj_reconstructed_tensor)
            gene_adj_reconstructed_np = gene_adj_reconstructed_tensor.cpu().numpy()
            print(f"Reconstructed gene adjacency matrix shape: {gene_adj_reconstructed_np.shape}")
            
            output_cell_path = os.path.join(output_path, "reconstructed_cell_adjacency.npy")
            output_gene_path = os.path.join(output_path, "reconstructed_gene_adjacency.npy")
            
            np.save(output_cell_path, adj_reconstructed_np)
            np.save(output_gene_path, gene_adj_reconstructed_np)
            
            print(f"Saved reconstructed cell adjacency matrix to: {output_cell_path}")
            print(f"Saved reconstructed gene adjacency matrix to: {output_gene_path}")
            
            return adj_reconstructed_np, gene_adj_reconstructed_np
        else:
            print("Error: Model does not have decoder.forward_all method")
            return None, None

if __name__ == "__main__":
    main()