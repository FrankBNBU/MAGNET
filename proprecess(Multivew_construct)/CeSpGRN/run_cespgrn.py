import sys, os
sys.path.append('./src/')
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import g_admm as CeSpGRN
import kernel
import warnings
warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from umap import UMAP

def preprocess(counts):
    """预处理数据"""
    # 根据文库大小进行归一化
    libsize = np.median(np.sum(counts, axis=1))
    counts = counts / np.sum(counts, axis=1)[:,None] * libsize
    # 进行log转换
    counts = np.log1p(counts)
    return counts

def main():
    # 1. 读取数据
    print("正在读取数据...")
    # 请替换为您的数据路径
    path = "./data/COUNTS-THP-1/"
    counts = pd.read_csv(path + "counts.csv", index_col=0).values
    annotation = pd.read_csv(path + "anno.csv", index_col=0)
    
    print(f"数据形状: {counts.shape}")
    
    # 2. 数据预处理
    print("正在预处理数据...")
    counts = preprocess(counts)
    
    # 3. 降维
    print("正在进行PCA降维...")
    pca_op = PCA(n_components=20)
    X_pca = pca_op.fit_transform(counts)
    
    # 4. 设置参数
    print("设置参数...")
    bandwidth = 0.1  # 可以调整
    n_neigh = 30    # 可以调整
    lamb = 0.1      # 可以调整
    max_iters = 1000
    batchsize = 120
    
    # 5. 计算核函数
    print("计算核函数...")
    start_time = time.time()
    K, K_trun = kernel.calc_kernel_neigh(X_pca, k=5, bandwidth=bandwidth, 
                                        truncate=True, truncate_param=n_neigh)
    print(f"考虑的邻居数量: {np.sum(K_trun[int(counts.shape[0]/2), :] > 0)}")
    
    # 6. 估计协方差矩阵
    print("估计协方差矩阵...")
    empir_cov = CeSpGRN.est_cov(X=counts, K_trun=K_trun, weighted_kt=True)
    
    # 7. 估计细胞特异性GRN
    print("估计细胞特异性GRN...")
    cespgrn = CeSpGRN.G_admm_minibatch(X=counts[:, None, :], K=K, 
                                      pre_cov=empir_cov, batchsize=batchsize)
    thetas = cespgrn.train(max_iters=max_iters, n_intervals=100, lamb=lamb)
    
    # 8. 保存结果
    print("保存结果...")
    output_file = f"./thetas_{bandwidth}_{lamb}_{n_neigh}.npy"
    np.save(file=output_file, arr=thetas)
    print(f"结果已保存到: {output_file}")
    print(f"总计算时间: {time.time() - start_time:.2f} 秒")
    
    # 9. 可视化
    print("生成可视化结果...")
    thetas = thetas.reshape(thetas.shape[0], -1)
    thetas_pca = pca_op.fit_transform(thetas)
    
    umap_op = UMAP(n_components=2, min_dist=0.8, random_state=0)
    thetas_umap = umap_op.fit_transform(thetas)
    
    # 保存PCA图
    plt.figure(figsize=(10,7))
    for i in np.sort(np.unique(annotation.values.squeeze())):
        idx = np.where(annotation.values.squeeze() == i)
        plt.scatter(thetas_pca[idx, 0], thetas_pca[idx, 1], label=i, s=10)
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, markerscale=3)
    plt.title(f"bandwidth: {bandwidth}, n_neigh: {n_neigh}, lamb: {lamb}")
    plt.savefig(f"plot_thetas_{bandwidth}_{lamb}_{n_neigh}_pca.png", bbox_inches="tight")
    
    # 保存UMAP图
    plt.figure(figsize=(10,7))
    for i in np.sort(np.unique(annotation.values.squeeze())):
        idx = np.where(annotation.values.squeeze() == i)
        plt.scatter(thetas_umap[idx, 0], thetas_umap[idx, 1], label=i, s=10)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, markerscale=3)
    plt.title(f"bandwidth: {bandwidth}, n_neigh: {n_neigh}, lamb: {lamb}")
    plt.savefig(f"plot_thetas_{bandwidth}_{lamb}_{n_neigh}_umap.png", bbox_inches="tight")
    
    print("可视化结果已保存")

if __name__ == "__main__":
    main() 