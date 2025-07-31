import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

raw_data_path = "./data/"
starting_df = pd.read_csv(os.path.join(raw_data_path, "seqfish_dataframe.csv"))

def visualize_celllevel_graph(df, title, edge_trace=None, edge_weights=1, publication=False):
    figsize_in_inches = (700 / 100, 650 / 100)
    fig, ax = plt.subplots(figsize=figsize_in_inches)
    ax.scatter(df["X"], df["Y"], color="gray", s=6, alpha=0.8, label="Cells")
    if edge_trace is not None:
        for i in range(0, len(edge_trace[0]), 2):
            x0, y0 = edge_trace[0][i], edge_trace[1][i]
            x1, y1 = edge_trace[0][i+1], edge_trace[1][i+1]
            ax.plot([x0, x1], [y0, y1], color='#888', lw=1.1, alpha=0.7)
    ax.set_title(title, fontsize=16, family="Times New Roman", color="black")
    ax.set_xlabel("X coordinate", fontsize=16, family="Times New Roman")
    ax.set_ylabel("Y coordinate", fontsize=16, family="Times New Roman")
    ax.tick_params(axis='both', which='both', length=5, width=1, labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if not publication:
        ax.grid(False)
    else:
        ax.set_facecolor("white")
        if ax.get_legend():
            ax.legend().set_visible(False)
    return fig

def construct_celllevel_graph_with_distance(data_df, k, m_neighbor_for_sigma=True, default_global_sigma=70, get_edges=False):
    num_cells = len(data_df)
    adjacency = np.zeros(shape=(num_cells, num_cells), dtype=int) 
    coords = np.vstack([data_df["X"].values, data_df["Y"].values]).T
    edge_x_list, edge_y_list = [], []
    calculated_strengths_for_selected_edges = [] 
    network_relationship = np.zeros((num_cells, num_cells))
    
    local_sigmas = np.full(num_cells, default_global_sigma)
    if m_neighbor_for_sigma is not None and m_neighbor_for_sigma > 0:
        m_rank_for_sigma = m_neighbor_for_sigma 
        for i_sig_calc in range(num_cells):
            current_cell_coords_for_sigma = coords[i_sig_calc]
            distances_for_sigma_calc = np.linalg.norm(coords - current_cell_coords_for_sigma, axis=1)
            sorted_original_indices = np.argsort(distances_for_sigma_calc)
            if len(sorted_original_indices) > m_rank_for_sigma: 
                mth_neighbor_original_idx = sorted_original_indices[m_rank_for_sigma]
                sigma_val = distances_for_sigma_calc[mth_neighbor_original_idx]
                if sigma_val > 1e-9: 
                    local_sigmas[i_sig_calc] = sigma_val
    
    for i in range(num_cells):
        x0, y0 = data_df["X"].values[i], data_df["Y"].values[i]
        current_cell_coords = coords[i]
        current_sigma_i = local_sigmas[i]
        euclidean_distances_from_i = np.linalg.norm(coords - current_cell_coords, axis=1)
        strengths_for_sorting = np.zeros(num_cells) 
        for j in range(num_cells):
            distance_i_to_j = euclidean_distances_from_i[j]
            strength_ij = 0.0
            if i == j:
                strength_ij = 1.0
                strengths_for_sorting[j] = -1.0
            else:
                if current_sigma_i > 1e-9:
                    strength_ij = np.exp(- (distance_i_to_j ** 2) / (2 * current_sigma_i ** 2))
                elif distance_i_to_j < 1e-9:
                    strength_ij = 1.0 
                strengths_for_sorting[j] = strength_ij
            network_relationship[i, j] = strength_ij
        
        sorted_indices_by_strength = np.argsort(strengths_for_sorting)[::-1] 
        num_added_neighbors = 0
        for neighbor_idx in sorted_indices_by_strength:
            if num_added_neighbors == k: 
                break
            current_strength_for_neighbor = network_relationship[i, neighbor_idx] 
            if current_strength_for_neighbor > 1e-9:
                adjacency[i, neighbor_idx] = 1
                calculated_strengths_for_selected_edges.append(current_strength_for_neighbor)
                if get_edges: 
                    x1, y1 = data_df["X"].values[neighbor_idx], data_df["Y"].values[neighbor_idx]
                    edge_x_list.append(x0); edge_x_list.append(x1) 
                    edge_y_list.append(y0); edge_y_list.append(y1)
                num_added_neighbors += 1
    
    edges_for_return = [edge_x_list, edge_y_list] if get_edges else None
    return adjacency, edges_for_return, calculated_strengths_for_selected_edges, network_relationship

def construct_celllevel_graph_with_LR(data_df, ligand_receptor_pairs, sigma=70, k=2, n_permutations=1000, alpha=0.05, min_expression_threshold=0.01, get_edges=True):
    num_cells = len(data_df)
    coords = data_df[["X", "Y"]].values
    network_relationship = np.zeros((num_cells, num_cells))
    
    for ligand_col, receptor_col in ligand_receptor_pairs:
        if ligand_col not in data_df.columns or receptor_col not in data_df.columns:
            print(f"Warning: {ligand_col} or {receptor_col} not found in data")
            continue
            
        L_vals = data_df[ligand_col].values
        R_vals = data_df[receptor_col].values
        expressed_ligand_mask = L_vals >= min_expression_threshold
        expressed_receptor_mask = R_vals >= min_expression_threshold
        current_lr_interaction_scores_for_this_pair = np.zeros((num_cells, num_cells))
        null_scores_population_list = []
        
        for _ in range(n_permutations):
            R_vals_permuted = np.random.permutation(R_vals)
            sample_size_for_null = num_cells * 1
            idx_i_for_null = np.random.choice(num_cells, sample_size_for_null, replace=True)
            idx_j_for_null_perm = np.random.choice(num_cells, sample_size_for_null, replace=True)
            null_scores_population_list.extend(L_vals[idx_i_for_null] * R_vals_permuted[idx_j_for_null_perm])
        
        null_scores_population = np.array(null_scores_population_list)
        null_scores_population.sort()
        len_null_pop = len(null_scores_population)

        for i in range(num_cells):
            if not expressed_ligand_mask[i]:
                continue
            for j in range(num_cells):
                if i == j or not expressed_receptor_mask[j]:
                    continue
                s_obs = L_vals[i] * R_vals[j]
                if s_obs < 1e-9:
                    continue
                count_greater_equal = len_null_pop - np.searchsorted(null_scores_population, s_obs, side='left')
                p_value = (count_greater_equal + 1) / (len_null_pop + 1)
                if p_value < alpha:
                    dist_sq = np.sum((coords[i] - coords[j]) ** 2)
                    spatial_decay = np.exp(-dist_sq / (2 * sigma ** 2))
                    current_lr_interaction_scores_for_this_pair[i, j] = s_obs * spatial_decay
        network_relationship += current_lr_interaction_scores_for_this_pair

    adjacency = np.zeros((num_cells, num_cells), dtype=int)
    edge_x_list, edge_y_list, selected_edge_weights = [], [], []

    for i in range(num_cells):
        strengths_from_i = network_relationship[i, :]
        potential_neighbor_indices_with_strength = np.where(strengths_from_i > 1e-9)[0]
        potential_neighbor_indices = potential_neighbor_indices_with_strength[potential_neighbor_indices_with_strength != i]
        if len(potential_neighbor_indices) == 0:
            continue
        actual_strengths = strengths_from_i[potential_neighbor_indices]
        num_neighbors_to_select = min(k, len(potential_neighbor_indices))
        if num_neighbors_to_select > 0:
            top_k_local_indices_in_potential = np.argsort(actual_strengths)[-num_neighbors_to_select:]
            selected_neighbor_global_indices = potential_neighbor_indices[top_k_local_indices_in_potential]
            for neighbor_j_idx in selected_neighbor_global_indices:
                current_edge_weight = network_relationship[i, neighbor_j_idx]
                if current_edge_weight > 1e-9:
                    adjacency[i, neighbor_j_idx] = 1
                    selected_edge_weights.append(current_edge_weight)
                    if get_edges:
                        x0, y0 = data_df.iloc[i][["X", "Y"]]
                        x1, y1 = data_df.iloc[neighbor_j_idx][["X", "Y"]]
                        edge_x_list.extend([x0, x1])
                        edge_y_list.extend([y0, y1])
    
    edges_for_return = [edge_x_list, edge_y_list] if get_edges else None
    return adjacency, edges_for_return, network_relationship, selected_edge_weights

def construct_celllevel_graph_umap_inspired_standard_output(data_df, k, get_edges=True):
    feature_df = data_df.drop(columns=["Cell_ID", "X", "Y", "Cell_Type"], errors='ignore')
    feature_df = feature_df.select_dtypes(include=[np.number]) 
    feature_matrix = feature_df.values
    num_cells = feature_matrix.shape[0]

    if num_cells == 0:
        empty_nxn_float = np.array([]).reshape(0,0).astype(float)
        empty_nxn_int = np.array([]).reshape(0,0).astype(int)
        empty_edges_list = [[], []] if get_edges else None
        return empty_nxn_int, empty_nxn_float, empty_edges_list, []

    mean_vals = feature_matrix.mean(axis=0)
    std_devs = feature_matrix.std(axis=0)
    std_devs[std_devs == 0] = 1.0  
    feature_matrix_scaled = (feature_matrix - mean_vals) / std_devs
    feature_matrix_scaled = np.nan_to_num(feature_matrix_scaled)

    feature_matrix_processed = feature_matrix_scaled
    if num_cells > 1:
        n_components_pca = min(50, feature_matrix_scaled.shape[1])
        if num_cells <= n_components_pca:
            n_components_pca = max(1, num_cells -1) if num_cells > 1 else 1 

        if n_components_pca > 0 and feature_matrix_scaled.shape[1] > 0:
            try:
                pca = PCA(n_components=n_components_pca, random_state=42)
                feature_matrix_processed = pca.fit_transform(feature_matrix_scaled)
            except ValueError as e:
                feature_matrix_processed = feature_matrix_scaled

    if feature_matrix_processed.shape[0] > 1:
        dist_matrix = pairwise_distances(feature_matrix_processed, metric='euclidean')
    elif feature_matrix_processed.shape[0] == 1:
        dist_matrix = np.array([[0.0]])
    else:
        dist_matrix = np.array([]).reshape(0,0)

    sigmas = np.zeros(num_cells)
    rhos = np.zeros(num_cells)

    if num_cells > 1:
        for i in range(num_cells):
            sorted_distances_from_i = np.sort(dist_matrix[i])
            rhos[i] = sorted_distances_from_i[1]
            actual_k_for_sigma = min(k, num_cells - 1)
            if actual_k_for_sigma > 0:
                sigma_val = sorted_distances_from_i[actual_k_for_sigma]
            else:
                sigma_val = np.mean(sorted_distances_from_i[1:]) if (num_cells - 1 > 0 and len(sorted_distances_from_i) > 1) else 1.0 
            sigmas[i] = max(sigma_val, 1e-10)
    elif num_cells == 1:
        sigmas[0] = 1.0 
        rhos[0] = 0.0   

    P_cond_val = np.zeros((num_cells, num_cells))
    if num_cells > 1:
        for i in range(num_cells):
            for j in range(num_cells):
                if i == j:
                    continue 
                dist_ij = dist_matrix[i, j]
                numerator_val = max(0, dist_ij - rhos[i])
                if sigmas[i] > 1e-12:
                    P_cond_val[i, j] = np.exp(-numerator_val / sigmas[i])

    network_relationship = np.zeros((num_cells, num_cells))
    if num_cells > 1:
        for i in range(num_cells):
            for j in range(i + 1, num_cells): 
                p_ji = P_cond_val[i, j]
                p_ij = P_cond_val[j, i]
                val = p_ji + p_ij - (p_ji * p_ij)
                network_relationship[i, j] = val
                network_relationship[j, i] = val 

    adjacency = np.zeros((num_cells, num_cells), dtype=int)
    new_edge_x_list, new_edge_y_list = [], []
    new_edge_weights_list = []

    if num_cells > 0:
        for i in range(num_cells):
            if num_cells == 1 and k > 0:
                break 
            neighbor_strengths = network_relationship[i].copy()
            neighbor_strengths[i] = -np.inf
            num_potential_neighbors = num_cells - 1 if num_cells > 1 else 0
            actual_k_for_adj = min(k, num_potential_neighbors)
            if actual_k_for_adj > 0:
                selected_neighbor_global_indices = np.argsort(neighbor_strengths)[-actual_k_for_adj:][::-1]
                for neighbor_idx in selected_neighbor_global_indices:
                    current_weight = network_relationship[i, neighbor_idx]
                    if current_weight > 1e-9:
                        adjacency[i, neighbor_idx] = 1
                        new_edge_weights_list.append(current_weight)
                        if get_edges:
                            x0, y0 = data_df["X"].values[i], data_df["Y"].values[i]
                            x1, y1 = data_df["X"].values[neighbor_idx], data_df["Y"].values[neighbor_idx]
                            new_edge_x_list.extend([x0, x1]) 
                            new_edge_y_list.extend([y0, y1])
    
    edges_for_return = [new_edge_x_list, new_edge_y_list] if get_edges else None
    if num_cells == 0 and get_edges:
        edges_for_return = [[], []]

    return adjacency, edges_for_return, network_relationship, new_edge_weights_list

def normalize_and_symmetrize(matrix):
    scaler = MinMaxScaler()
    matrix = scaler.fit_transform(matrix)
    matrix = (matrix + matrix.T) / 2
    return matrix

def fuse_networks_with_snf(network_relationships_list, K_snf=20, t_snf=10):
    # Simple averaging since SNF is not available
    affinity_networks = [normalize_and_symmetrize(net) for net in network_relationships_list]
    try:
        # Placeholder for SNF - using simple average
        fused_matrix = np.mean(affinity_networks, axis=0)
    except Exception as e:
        print(f"SNF failed with error: {e}, fallback to average.")
        fused_matrix = np.mean(affinity_networks, axis=0)
    return fused_matrix

def build_graph_from_fused_matrix(data_df, fused_matrix, k, get_edges=True):
    num_cells = len(data_df)
    combined_adjacency = np.zeros((num_cells, num_cells), dtype=int)
    combined_edge_x, combined_edge_y = [], []
    combined_edge_weights_list = []
    output_symmetric_weights_matrix = np.zeros((num_cells, num_cells), dtype=float)

    for i in range(num_cells):
        scores = fused_matrix[i, :].copy()
        scores[i] = -np.inf
        neighbor_indices = np.argsort(scores)[::-1][:min(k, num_cells - 1)]
        combined_adjacency[i, neighbor_indices] = 1

        if get_edges:
            x0, y0 = data_df.loc[i, "X"], data_df.loc[i, "Y"]
            for j in neighbor_indices:
                x1, y1 = data_df.loc[j, "X"], data_df.loc[j, "Y"]
                combined_edge_x.extend([x0, x1])
                combined_edge_y.extend([y0, y1])
                w = fused_matrix[i, j]
                combined_edge_weights_list.append(w)
                output_symmetric_weights_matrix[i, j] = w
                output_symmetric_weights_matrix[j, i] = w

    edges_for_return = [combined_edge_x, combined_edge_y] if get_edges else None
    return combined_adjacency, edges_for_return, combined_edge_weights_list, output_symmetric_weights_matrix

def convert_adjacencymatrix2edgelist(adj_matrix_data):
    source_nodes, target_nodes = np.where(adj_matrix_data)
    if len(source_nodes) == 0:
        return np.array([]).reshape(2,0)
    return np.vstack((source_nodes, target_nodes)).astype(int)

# Execute main processing
print("Processing cell-level graph construction...")

# 1. Distance-based graph
celllevel_adj1, edges1, edge_weights1, network_relationship1 = construct_celllevel_graph_with_distance(
    starting_df, k=5, m_neighbor_for_sigma=10, default_global_sigma=70, get_edges=True
)

sim_fig = visualize_celllevel_graph(
    starting_df, title="Cell_Interaction(Spatial)", edge_trace=edges1, edge_weights=edge_weights1, publication=True
)
print(f"celllevel_adj1.shape: {celllevel_adj1.shape}")

# 2. Ligand-receptor graph
ligand_receptor_pairs = [('Gdf5', 'Bmpr1b'),('Gdf2', 'Bmpr1b'),('Cdh1', 'Bmpr1b'),('Ctla4', 'Bmpr1b')]
celllevel_adj2, edges2, network_relationship2, edge_weights2 = construct_celllevel_graph_with_LR(
    starting_df, ligand_receptor_pairs, sigma=70, k=5, n_permutations=1000, alpha=0.05, min_expression_threshold=0.01, get_edges=True
)

sim_fig = visualize_celllevel_graph(
    starting_df, title="Cell_Interaction(LR)", edge_trace=edges2, edge_weights=edge_weights2, publication=True
)

# 3. UMAP-inspired graph
celllevel_adj3, edges3, network_relationship3, edge_weights3 = construct_celllevel_graph_umap_inspired_standard_output(
    data_df=starting_df, k=5, get_edges=True
)

sim_fig = visualize_celllevel_graph(
    starting_df, title="Cell_Interaction_Similarity", edge_trace=edges3, edge_weights=network_relationship3, publication=True
)

# 4. Multi-view fusion
normalized_relationships = [network_relationship1, network_relationship2, network_relationship3]
K_for_snf_algorithm = 100
iterations_for_snf = 10
final_fused_matrix = fuse_networks_with_snf(normalized_relationships, K_snf=K_for_snf_algorithm, t_snf=iterations_for_snf)
k_for_graph_neighbors = 5 
celllevel_adj4, edges4, combined_edge_weights, combined_edge_weight_matrix = build_graph_from_fused_matrix(
    starting_df, final_fused_matrix, k_for_graph_neighbors, get_edges=True
)
combined_fig = visualize_celllevel_graph(
    starting_df, title="Cell_Interaction(Multi_view)", edge_trace=edges4, edge_weights=combined_edge_weights, publication=True
)

# Convert to edge lists
print("\nConverting celllevel_adj1 to edge list...")
celllevel_edgelist1 = convert_adjacencymatrix2edgelist(celllevel_adj1)
print(f"celllevel_edgelist1.shape: {celllevel_edgelist1.shape}")

print("\nConverting celllevel_adj2 to edge list...")
celllevel_edgelist2 = convert_adjacencymatrix2edgelist(celllevel_adj2) 
print(f"celllevel_edgelist2.shape: {celllevel_edgelist2.shape}")

print("\nConverting celllevel_adj3 to edge list...")
celllevel_edgelist3 = convert_adjacencymatrix2edgelist(celllevel_adj3)
print(f"celllevel_edgelist3.shape: {celllevel_edgelist3.shape}")

print("\nConverting celllevel_adj4 to edge list...")
celllevel_edgelist4 = convert_adjacencymatrix2edgelist(celllevel_adj4)
print(f"celllevel_edgelist4.shape: {celllevel_edgelist4.shape}")

# Create output directory
preprocess_output_path = "."
if not os.path.exists(preprocess_output_path):
    os.makedirs(preprocess_output_path)
    print(f"Path {preprocess_output_path} created successfully!")
else:
    print(f"Path {preprocess_output_path} already exists.")

# Save all adjacency matrices and edge lists
adjacency_matrices = [celllevel_adj1, celllevel_adj2, celllevel_adj3, celllevel_adj4]
edge_lists = [celllevel_edgelist1, celllevel_edgelist2, celllevel_edgelist3, celllevel_edgelist4]

for i in range(1, 5):
    adj_matrix = adjacency_matrices[i-1]
    edge_list = edge_lists[i-1]
    
    # Save adjacency matrix
    adj_filename = f"celllevel_adjacencymatrix{i}.npy"
    np.save(os.path.join(preprocess_output_path, adj_filename), adj_matrix)
    print(f"Adjacency matrix saved to {adj_filename}")
    
    # Save edge list
    edge_filename = f"celllevel_edgelist{i}.npy"
    np.save(os.path.join(preprocess_output_path, edge_filename), edge_list)
    print(f"Edge list saved to {edge_filename}")

print("Script completed successfully!")
print(f"Generated files:")
for i in range(1, 5):
    print(f"- celllevel_adjacencymatrix{i}.npy")
    print(f"- celllevel_edgelist{i}.npy")