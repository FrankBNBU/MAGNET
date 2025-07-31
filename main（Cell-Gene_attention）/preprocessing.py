import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import PCA
import time
from rich.progress import track
from rich.table import Table
from rich.console import Console
import random
import sys
from node2vec import Node2Vec
import os
sys.path.insert(0, os.path.abspath('./submodules/'))
sys.path.append('./submodules/CeSpGRN/src/')
sys.path.append('../')


def select_LRgenes(data_df, num_genespercell, lr_database = 2):
    '''
    Selects LR and relevant background genes to be included for GRN inference.
    
    :data_df: pd.DataFrame : represents the spatial data and contains the following columns ["Cell_ID", "X", "Y", "Cell_Type", "Gene 1", ..., "Gene n"]
    :num_genespercell: int : represents the number of genes to be included for each Cell-Specific GRN
    :lr_database: int: 0/1/2 corresponding to LR database (0: CellTalkDB Mouse,1: CellTalkDB Human, 2: scMultiSim Simulated)
    
    :return : pd.DataFrame with relevant gene columns preserved and dictionary mapping LR genes to their numerical ids
    '''
    if lr_database == 0:
        sample_counts = data_df.drop(["Cell_ID", "X", "Y", "Cell_Type"], axis = 1)
        # lr_df = pd.read_csv("../data/celltalk_human_lr_pair.txt", sep="\t")
        lr_df = pd.read_csv("../data/celltalk_mouse_lr_pair.txt", sep="\t")
        
        receptors = set(lr_df["receptor_gene_symbol"].str.upper().to_list())
        ligands = set(lr_df["ligand_gene_symbol"].str.upper().to_list())
        real2uppercase = {x:x.upper() for x in sample_counts.columns}
        uppercase2real = {upper:real for real,upper in real2uppercase.items()}
        candidate_genes = set(np.vectorize(real2uppercase.get)(sample_counts.columns.to_numpy()))
        
        selected_ligands = candidate_genes.intersection(ligands)
        selected_receptors = candidate_genes.intersection(receptors)
        selected_lrs = selected_ligands | selected_receptors
        
        if len(selected_lrs) > num_genespercell // 2 + 1:
            selected_lrs = set(random.sample(tuple(selected_lrs), num_genespercell // 2 + 1))
            selected_ligands = selected_lrs.intersection(selected_ligands)
            selected_receptors = selected_lrs.intersection(selected_receptors)
        
        print(f"Using {len(selected_ligands)} ligands and {len(selected_receptors)} receptors to be included in the {num_genespercell} selected genes per cell. \n")
        
        num_genesleft = num_genespercell - len(selected_ligands) - len(selected_receptors)
        
        candidate_genesleft = candidate_genes - selected_ligands - selected_receptors
        
        
        selected_randomgenes = set(random.sample(tuple(candidate_genesleft), num_genesleft))
        
        selected_genes = list(selected_randomgenes | selected_ligands | selected_receptors)
                
        selected_columns = ["Cell_ID", "X", "Y", "Cell_Type"] + np.vectorize(uppercase2real.get)(selected_genes).tolist()
        selected_df = data_df[selected_columns]

        
        console = Console()
        table = Table(show_header=True, header_style="bold")
        table.add_column("Selected Ligands", style="cyan")
        table.add_column("Selected Receptors", style="deep_pink3")
        table.add_column("Selected Random Genes", justify="right")
        table.add_row("\n".join(selected_ligands),"\n".join(selected_receptors),"\n".join(selected_randomgenes))
        console.print(table)
                
        lr2id = {gene:list(selected_df.columns).index(gene)-4 for gene in np.vectorize(uppercase2real.get)(list(selected_ligands|selected_receptors))}
        
        return selected_df, lr2id
    
    elif lr_database==2:
  
        sample_counts = data_df.drop(["Cell_ID", "X", "Y", "Cell_Type"], axis = 1)
        candidate_genes = sample_counts.columns.to_numpy()
        scmultisim_lrs = pd.read_csv("../data/scMultiSim/simulated/cci_gt.csv")[["ligand", "receptor"]]
        scmultisim_lrs["ligand"] = scmultisim_lrs["ligand"]
        scmultisim_lrs["receptor"] = scmultisim_lrs["receptor"]
        selected_ligands = np.unique(scmultisim_lrs["ligand"])
        selected_receptors = np.unique(scmultisim_lrs["receptor"])
        selected_lrs = np.concatenate((selected_ligands,selected_receptors),axis=0)

        # remove_index = np.where(selected_lrs == "gene3")
        # selected_lrs = np.delete(selected_lrs, remove_index)

        num_genesleft = num_genespercell - len(selected_ligands) - len(selected_receptors)
        indices  = np.argwhere(candidate_genes == selected_lrs)
        candidate_genesleft = np.delete(candidate_genes, indices)
        selected_randomgenes = random.sample(set(candidate_genesleft), num_genesleft)
        
        console = Console()
        table = Table(show_header=True, header_style="bold")
        table.add_column("Selected Ligands", style="cyan")
        table.add_column("Selected Receptors", style="deep_pink3")
        table.add_column("Selected Random Genes", justify="right")
        table.add_row("\n".join([str(x) for x in selected_ligands]),"\n".join([str(x) for x in selected_receptors]),"\n".join([str(x) for x in selected_randomgenes]))
        console.print(table)

        selected_genes = np.concatenate((selected_lrs, selected_randomgenes),axis=0)


        new_columns = ["Cell_ID", "X", "Y", "Cell_Type"] + list(selected_genes)
        selected_df = data_df[new_columns]
        lr2id = {gene:list(selected_df.columns).index(gene)-4 for gene in selected_genes}

        return selected_df, lr2id
        
    else:
        raise Exception("Invalid lr_database type")
    

def infer_initial_grns(data_df, cespgrn_hyperparams):
    
    '''
    Infers the starting cell specific GRNs with CeSpGRN submodule.
    
    :data_df: pd.DataFrame : represents the spatial data and contains the following columns ["Cell_ID", "X", "Y", "Cell_Type", "Gene 1", ..., "Gene n"]
    :cespgrn_hyperparams: dict(): dictionary of CeSpGRN hyperparameters
    
    :return: nd.array(shape=(numcells, numgenespercell, numgenespercell))
    '''
    console = Console()
    
    from submodules.CeSpGRN.src import kernel
    from submodules.CeSpGRN.src import g_admm as CeSpGRN
    
    with console.status("[cyan] Preparing CeSpGRN ...") as status:
        status.update(spinner="aesthetic", spinner_style="cyan")
        counts = data_df.drop(["Cell_ID", "X", "Y", "Cell_Type"], axis = 1).values
        # print(f"GRNs are dimension ({counts.shape[1]} by {counts.shape[1]}) for each of the {counts.shape[0]} cells\n")
        
        # Normalize Counts??? 
        pca_op = PCA(n_components = 20)
        X_pca = pca_op.fit_transform(counts)

        # hyper-parameters
        bandwidth = cespgrn_hyperparams["bandwidth"]
        n_neigh = cespgrn_hyperparams["n_neigh"]
        lamb = cespgrn_hyperparams["lamb"]
        max_iters = cespgrn_hyperparams["max_iters"]

        # calculate the kernel function
        status.update(status="[cyan] calculating kernel function ...")
        K, K_trun = kernel.calc_kernel_neigh(X_pca, k = 5, bandwidth = bandwidth, truncate = True, truncate_param = n_neigh)

        # estimate covariance matrix, output is empir_cov of the shape (ncells, ngenes, ngenes)
        status.update(status="[cyan] estimating covariance matrix ...")
        empir_cov = CeSpGRN.est_cov(X = counts, K_trun = K_trun, weighted_kt = True)

        # estimate cell-specific GRNs
        status.update(status="[cyan] Load in CeSpGRN model ...")
        cespgrn = CeSpGRN.G_admm_minibatch(X=counts[:, None, :], K=K, pre_cov=empir_cov, batchsize = 120)
        
        status.update(status="[cyan] Ready to train ...")
        time.sleep(3)
        
    grns = cespgrn.train(max_iters=max_iters, n_intervals=100, lamb=lamb)
    
    return grns




def construct_genelevel_graph(disjoint_grns, node_type = "int", lrgenes = None):
    '''
    Constructs gene level graph from cell-specific GRNs.
    
    :disjoint_grns: nd.array(shape=(numcells, numgenes, numgenes)) : np array of cell-specific GRNs (output of CeSpGRN)
    :node_type: str: Either "int" or "str" to set for the node labels
    :lrgenes: list: List of ligand-receptor gene indices
    
    :return: nx.Graph object (gene level), mapping between integer node names to gene names, reverse mapping
    '''
    
    numgenes = disjoint_grns[0].shape[0]
    numcells = disjoint_grns.shape[0]
    num2gene = {}
    gene2num = {}

    if lrgenes is not None:
        assert max(lrgenes) <= numgenes

    grn_graph_list = []
    for cellnum, grn in enumerate(track(disjoint_grns, description=f"[cyan]Combining individual GRNs")):
        G = nx.from_numpy_matrix(grn)
        grn_graph_list.append(G)
        for i in range(numgenes):
            num2gene[cellnum*numgenes+i] = f"Cell{cellnum}_Gene{i}"
            gene2num[f"Cell{cellnum}_Gene{i}"] = cellnum * numgenes + i

    union_of_grns = nx.disjoint_union_all(grn_graph_list)
    gene_level_graph = nx.relabel_nodes(union_of_grns, num2gene)

    if node_type == "str":
        gene_level_graph = gene_level_graph
    elif node_type == "int":
        gene_level_graph = nx.convert_node_labels_to_integers(gene_level_graph)

    assert len(gene_level_graph.nodes()) == numcells * numgenes

    return gene_level_graph, num2gene, gene2num, union_of_grns
        


def get_gene_features(graph, type="node2vec"):
    
    console = Console()
    console.print(f"[cyan]4. Constructing {type} Gene Level Features")
    
    if type == "node2vec":
        node2vec = Node2Vec(graph, dimensions=64, walk_length=15, num_walks=100, workers=4)
        model = node2vec.fit()

        gene_feature_vectors = model.wv.vectors
        
    return gene_feature_vectors, model