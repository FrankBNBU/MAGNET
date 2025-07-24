import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy import linalg
import matplotlib.pyplot as plt
import seaborn as sns

def cluster_expression_matrix(counts, cluster_labels):
    """将表达矩阵按cluster分组"""
    unique_clusters = np.unique(cluster_labels)
    cluster_matrices = {}
    for cluster in unique_clusters:
        cluster_idx = np.where(cluster_labels == cluster)[0]
        cluster_matrices[cluster] = counts[cluster_idx, :]
    return cluster_matrices

def find_cluster_connection(clusterA_matrix, clusterB_matrix):
    """计算两个cluster之间的连接向量"""
    # 计算每个cluster的平均表达
    mean_A = np.mean(clusterA_matrix, axis=0)
    mean_B = np.mean(clusterB_matrix, axis=0)
    
    # 使用最小二乘法求解连接向量
    # 求解方程：mean_A * x = mean_B
    connection_vector = np.linalg.lstsq(mean_A.reshape(-1, 1), mean_B, rcond=None)[0]
    
    return connection_vector

def calculate_grn_difference(grn_A, grn_B):
    """计算两个GRN的差异"""
    # 计算平均GRN
    mean_grn_A = np.mean(grn_A, axis=0)  # (gene, gene)
    mean_grn_B = np.mean(grn_B, axis=0)  # (gene, gene)
    
    # 计算差异
    grn_diff = mean_grn_A - mean_grn_B
    
    return grn_diff

def analyze_clusters(counts, cluster_labels, thetas):
    """主分析函数"""
    # 1. 按cluster分组表达矩阵
    cluster_matrices = cluster_expression_matrix(counts, cluster_labels)
    
    # 2. 按cluster分组GRN
    cluster_grns = {}
    for cluster in np.unique(cluster_labels):
        cluster_idx = np.where(cluster_labels == cluster)[0]
        cluster_grns[cluster] = thetas[cluster_idx, :, :]
    
    # 3. 分析结果存储
    results = {
        'cluster_connections': {},
        'grn_differences': {},
        'top_genes': {}
    }
    
    # 4. 计算每对cluster之间的关系
    clusters = list(cluster_matrices.keys())
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            clusterA = clusters[i]
            clusterB = clusters[j]
            
            # 计算cluster间连接向量
            connection = find_cluster_connection(
                cluster_matrices[clusterA],
                cluster_matrices[clusterB]
            )
            results['cluster_connections'][f'{clusterA}_to_{clusterB}'] = connection
            
            # 计算GRN差异
            grn_diff = calculate_grn_difference(
                cluster_grns[clusterA],
                cluster_grns[clusterB]
            )
            results['grn_differences'][f'{clusterA}_vs_{clusterB}'] = grn_diff
            
            # 找出最重要的基因
            top_genes = np.argsort(np.abs(connection))[-10:]  # 取前10个基因
            results['top_genes'][f'{clusterA}_to_{clusterB}'] = top_genes
    
    return results

def visualize_results(results, gene_names):
    """可视化分析结果"""
    # 1. 绘制cluster连接热图
    for connection_name, connection in results['cluster_connections'].items():
        plt.figure(figsize=(12, 4))
        plt.bar(range(len(connection)), connection)
        plt.title(f'Connection Vector: {connection_name}')
        plt.xlabel('Genes')
        plt.ylabel('Connection Strength')
        plt.savefig(f'connection_{connection_name}.png')
        plt.close()
    
    # 2. 绘制GRN差异热图
    for diff_name, diff in results['grn_differences'].items():
        plt.figure(figsize=(10, 8))
        sns.heatmap(diff, cmap='RdBu_r', center=0)
        plt.title(f'GRN Difference: {diff_name}') 
        plt.savefig(f'grn_diff_{diff_name}.png')
        plt.close()
    
    # 3. 输出重要基因
    for connection_name, top_genes in results['top_genes'].items():
        print(f"\nTop genes for {connection_name}:")
        for gene_idx in top_genes:
            print(f"{gene_names[gene_idx]}: {results['cluster_connections'][connection_name][gene_idx]:.3f}")

def main():
    # 1. 读取数据
    print("读取数据...")
    counts = pd.read_csv("counts.csv", index_col=0).values
    gene_names = pd.read_csv("counts.csv", index_col=0).columns
    thetas = np.load("thetas.npy")  # 之前计算的GRN结果
    
    # 2. 聚类（如果还没有聚类结果）
    print("进行聚类...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(counts)
    
    # 3. 分析
    print("开始分析...")
    results = analyze_clusters(counts, cluster_labels, thetas)
    
    # 4. 可视化
    print("生成可视化结果...")
    visualize_results(results, gene_names)
    
    print("分析完成！")

if __name__ == "__main__":
    main() 