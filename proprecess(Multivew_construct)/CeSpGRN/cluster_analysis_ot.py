import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import mutual_info_score
import ot
from ot.bregman import sinkhorn

def compute_mutual_information(clusterA_matrix, clusterB_matrix):
    """计算两个cluster之间的互信息"""
    mi_scores = []
    for gene in range(clusterA_matrix.shape[1]):
        # 将连续值离散化为bins
        bins = np.linspace(
            min(clusterA_matrix[:, gene].min(), clusterB_matrix[:, gene].min()),
            max(clusterA_matrix[:, gene].max(), clusterB_matrix[:, gene].max()),
            20
        )
        hist_A = np.histogram(clusterA_matrix[:, gene], bins=bins)[0]
        hist_B = np.histogram(clusterB_matrix[:, gene], bins=bins)[0]
        
        # 计算互信息
        mi = mutual_info_score(hist_A, hist_B)
        mi_scores.append(mi)
    
    return np.array(mi_scores)

def compute_optimal_transport(clusterA_matrix, clusterB_matrix, epsilon=0.01):
    """使用Sinkhorn算法计算最优传输"""
    # 计算成本矩阵（欧氏距离）
    cost_matrix = ot.dist(clusterA_matrix, clusterB_matrix)
    
    # 计算传输计划
    transport_plan = sinkhorn(
        np.ones(clusterA_matrix.shape[0]) / clusterA_matrix.shape[0],
        np.ones(clusterB_matrix.shape[0]) / clusterB_matrix.shape[0],
        cost_matrix,
        epsilon
    )
    
    return transport_plan

def compute_cluster_differences_ot(clusterA_matrix, clusterB_matrix):
    """使用OT方法计算cluster差异"""
    # 1. 计算传输计划
    transport_plan = compute_optimal_transport(clusterA_matrix, clusterB_matrix)
    
    # 2. 计算B的"A-view"表达
    B_A_view = transport_plan.T @ clusterA_matrix
    
    # 3. 计算差异向量
    diff_vectors = clusterB_matrix - B_A_view
    
    # 4. 计算每个基因的平均差异
    mean_diff = np.mean(diff_vectors, axis=0)
    
    # 5. 计算差异的显著性
    significance = np.std(diff_vectors, axis=0)
    
    return {
        'mean_diff': mean_diff,
        'significance': significance,
        'transport_plan': transport_plan,
        'B_A_view': B_A_view
    }

def analyze_clusters_advanced(counts, cluster_labels, thetas):
    """使用高级方法分析cluster"""
    # 1. 按cluster分组
    cluster_matrices = {}
    cluster_grns = {}
    for cluster in np.unique(cluster_labels):
        cluster_idx = np.where(cluster_labels == cluster)[0]
        cluster_matrices[cluster] = counts[cluster_idx, :]
        cluster_grns[cluster] = thetas[cluster_idx, :, :]
    
    results = {
        'ot_differences': {},
        'mi_scores': {},
        'transport_plans': {},
        'top_genes': {}
    }
    
    # 2. 分析每对cluster
    clusters = list(cluster_matrices.keys())
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            clusterA = clusters[i]
            clusterB = clusters[j]
            
            # 计算OT差异
            ot_results = compute_cluster_differences_ot(
                cluster_matrices[clusterA],
                cluster_matrices[clusterB]
            )
            results['ot_differences'][f'{clusterA}_vs_{clusterB}'] = ot_results
            
            # 计算互信息
            mi_scores = compute_mutual_information(
                cluster_matrices[clusterA],
                cluster_matrices[clusterB]
            )
            results['mi_scores'][f'{clusterA}_vs_{clusterB}'] = mi_scores
            
            # 找出最重要的基因（基于OT差异和互信息的组合）
            combined_scores = np.abs(ot_results['mean_diff']) * mi_scores
            top_genes = np.argsort(combined_scores)[-10:]
            results['top_genes'][f'{clusterA}_vs_{clusterB}'] = top_genes
    
    return results

def visualize_advanced_results(results, gene_names):
    """可视化高级分析结果"""
    # 1. 绘制OT差异图
    for diff_name, diff_results in results['ot_differences'].items():
        plt.figure(figsize=(12, 4))
        plt.bar(range(len(diff_results['mean_diff'])), 
                diff_results['mean_diff'],
                yerr=diff_results['significance'])
        plt.title(f'OT-based Differences: {diff_name}')
        plt.xlabel('Genes')
        plt.ylabel('Difference Score')
        plt.savefig(f'ot_diff_{diff_name}.png')
        plt.close()
    
    # 2. 绘制互信息热图
    for mi_name, mi_scores in results['mi_scores'].items():
        plt.figure(figsize=(12, 4))
        plt.bar(range(len(mi_scores)), mi_scores)
        plt.title(f'Mutual Information: {mi_name}')
        plt.xlabel('Genes')
        plt.ylabel('MI Score')
        plt.savefig(f'mi_{mi_name}.png')
        plt.close()
    
    # 3. 绘制传输计划热图
    for diff_name, diff_results in results['ot_differences'].items():
        plt.figure(figsize=(10, 8))
        sns.heatmap(diff_results['transport_plan'], cmap='YlOrRd')
        plt.title(f'Transport Plan: {diff_name}')
        plt.savefig(f'transport_{diff_name}.png')
        plt.close()
    
    # 4. 输出重要基因
    for diff_name, top_genes in results['top_genes'].items():
        print(f"\nTop genes for {diff_name}:")
        for gene_idx in top_genes:
            print(f"{gene_names[gene_idx]}: "
                  f"OT_diff={results['ot_differences'][diff_name]['mean_diff'][gene_idx]:.3f}, "
                  f"MI={results['mi_scores'][diff_name][gene_idx]:.3f}")

def main():
    # 1. 读取数据
    print("读取数据...")
    counts = pd.read_csv("counts.csv", index_col=0).values
    gene_names = pd.read_csv("counts.csv", index_col=0).columns
    thetas = np.load("thetas.npy")
    
    # 2. 聚类
    print("进行聚类...")
    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(counts)
    
    # 3. 高级分析
    print("开始高级分析...")
    results = analyze_clusters_advanced(counts, cluster_labels, thetas)
    
    # 4. 可视化
    print("生成可视化结果...")
    visualize_advanced_results(results, gene_names)
    
    print("分析完成！")

if __name__ == "__main__":
    main() 