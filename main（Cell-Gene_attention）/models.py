import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import pandas as pd
import math
from copy import deepcopy

class ContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.1, hard_negative_weight=1.0, use_dynamic_temp=False):
        super(ContrastiveLoss, self).__init__()
        self.base_temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.use_dynamic_temp = use_dynamic_temp
        
        # Allow temperature to be a tunable, non-learnable parameter
        if use_dynamic_temp:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.temperature = temperature
        
    def forward(self, features_1, features_2, hard_negatives=None):

        batch_size = features_1.shape[0]
        if features_1.shape != features_2.shape:
            raise ValueError(f"特征维度不匹配: {features_1.shape} vs {features_2.shape}")
            

        features_1 = F.normalize(features_1, p=2, dim=1)
        features_2 = F.normalize(features_2, p=2, dim=1)
        
        # 获取当前温度值
        current_temp = self.temperature if self.use_dynamic_temp else self.base_temperature
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features_1, features_2.T) / current_temp
        
        # 正样本对的索引
        positive_samples = torch.arange(batch_size).to(features_1.device)
        
        # 基础对比损失
        loss_1 = F.cross_entropy(similarity_matrix, positive_samples)
        loss_2 = F.cross_entropy(similarity_matrix.T, positive_samples)
        base_loss = (loss_1 + loss_2) / 2.0
        
        # 硬负样本损失（如果提供）
        hard_loss = 0.0
        if hard_negatives is not None:
            hard_negatives = F.normalize(hard_negatives, p=2, dim=1)
            
            # 计算与硬负样本的相似度
            hard_sim_1 = torch.matmul(features_1, hard_negatives.T) / current_temp
            hard_sim_2 = torch.matmul(features_2, hard_negatives.T) / current_temp

            hard_labels = torch.full((batch_size,), -1).to(features_1.device)

            pos_sim_1 = torch.diag(similarity_matrix)
            pos_sim_2 = torch.diag(similarity_matrix.T)
            
            hard_loss_1 = F.margin_ranking_loss(
                pos_sim_1.unsqueeze(1), hard_sim_1, 
                torch.ones_like(hard_sim_1), margin=0.1
            )
            hard_loss_2 = F.margin_ranking_loss(
                pos_sim_2.unsqueeze(1), hard_sim_2,
                torch.ones_like(hard_sim_2), margin=0.1
            )
            
            hard_loss = (hard_loss_1 + hard_loss_2) / 2.0 * self.hard_negative_weight
        
        return base_loss + hard_loss

class MultiviewEncoder(torch.nn.Module):

    def __init__(self, SubgraphEncoder, GraphEncoder, dropout=0.2,
                 contrastive_temp=0.2, contrastive_hard_weight=0.8):
        super(MultiviewEncoder, self).__init__()
        self.encoder_g = SubgraphEncoder
        self.encoder_c = GraphEncoder

        self.cell_hidden_dim = GraphEncoder.conv3.out_channels
        self.gene_hidden_dim = 32

        # Enhanced attention mechanism
        self.attention = AttentionalGNN(
            input_dim=self.cell_hidden_dim + self.gene_hidden_dim,
            hidden_dim=self.cell_hidden_dim,
            output_dim=self.cell_hidden_dim,
            dropout=dropout,
            num_heads=8
        )

        self.lambda_center = 0.1
        self.lambda_structure = 0.1

        self.alignment_loss = None  
        self.contrastive_loss = ContrastiveLoss(
            temperature=contrastive_temp,
            hard_negative_weight=contrastive_hard_weight,
            use_dynamic_temp=False
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(self.cell_hidden_dim * 3, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )

        self.final_projection = nn.Sequential(
            nn.Linear(self.cell_hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, self.cell_hidden_dim)
        )

        self.loss_weights = {
            'contrastive': 0.2,
            'alignment': 0.2,
            'orthogonality': 0.05,
        }

        self.adaptive_scaling = nn.Parameter(torch.tensor(1.0))

        self._initialize_weights()
        
    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _initialize_alignment_loss(self, xc_dim, device):

        if self.alignment_loss is None:

            self.alignment_loss = AlignmentLoss(
                z4_dim=self.cell_hidden_dim,
                xc_dim=xc_dim,
                lambda_center=self.lambda_center,
                lambda_structure=self.lambda_structure
            ).to(device)
    
    def _generate_hard_negatives(self, Z_c, Z_g, num_negatives=8):

        batch_size = Z_c.shape[0]
        
        if batch_size < num_negatives:
            return None

        device = Z_c.device # Get device from an input tensor
            
        # 归一化特征以计算更稳定的相似度
        Z_c_norm = F.normalize(Z_c, p=2, dim=1)
        Z_g_norm = F.normalize(Z_g, p=2, dim=1)
        
        # 计算cross-modal相似度矩阵 (cell vs gene)
        cross_similarity = torch.matmul(Z_c_norm, Z_g_norm.T)
        
        # 掩盖对角线（正样本对）
        cross_similarity.fill_diagonal_(-float('inf'))
        
        # 选择最相似的负样本作为硬负样本（这些是最容易混淆的）
        _, hard_indices = torch.topk(cross_similarity, k=min(num_negatives, batch_size-1), dim=1)
        
        # 随机选择一个硬负样本索引
        # BUG FIX: Ensure newly created tensors are on the same device as the model.
        selected_idx = torch.randint(0, hard_indices.shape[1], (batch_size,), device=device)
        hard_neg_indices = hard_indices[torch.arange(batch_size, device=device), selected_idx]
        
        return Z_g[hard_neg_indices]
    
    def _compute_orthogonality_loss(self, Z_c, Z_g):
        """
        计算特征正交性损失，进一步增强cell-level和gene-level特征的独立性
        """
        # 归一化特征
        Z_c_norm = F.normalize(Z_c, p=2, dim=1)
        Z_g_norm = F.normalize(Z_g, p=2, dim=1)
        
        # 计算平均特征向量
        mean_c = torch.mean(Z_c_norm, dim=0)
        mean_g = torch.mean(Z_g_norm, dim=0)
        
        # 正交性损失：希望两个模态的平均特征尽可能正交
        orthogonality_loss = torch.abs(torch.dot(mean_c, mean_g))
        
        return orthogonality_loss
        
    def forward(self, x_c, x_g, edge_index_c, edge_index_g, edge_index_c1, edge_index_c2, x_g_edge):
        # 动态初始化对齐损失（如果还没有初始化）
        if self.alignment_loss is None:
            self._initialize_alignment_loss(x_c.shape[1], x_c.device)
        
        # 第一步：GCN编码器提取特征
        Z_g, gene_embeddings = self.encoder_g(x_g, edge_index_g)
        Z_c = self.encoder_c(x_c, edge_index_c)
        Z_c1 = self.encoder_c(x_c, edge_index_c1)
        Z_c2 = self.encoder_c(x_c, edge_index_c2)
        
        # Step 2: Contrastive Learning
        hard_negatives = self._generate_hard_negatives(Z_c, Z_g)
        contrast_loss_1 = self.contrastive_loss(Z_c, Z_g, hard_negatives)
        contrast_loss_2 = self.contrastive_loss(Z_c1, Z_g, hard_negatives)
        contrast_loss_3 = self.contrastive_loss(Z_c2, Z_g, hard_negatives)
        orthogonality_loss = self._compute_orthogonality_loss(Z_c, Z_g)
        contrast_loss = contrast_loss_1 + contrast_loss_2 + contrast_loss_3 + self.loss_weights['orthogonality'] * orthogonality_loss
        
        # Step 3: Attention-based fusion
        Z = torch.cat((Z_c, Z_g), dim=1)
        Z1, _ = self.attention(Z, Z_g, Z_c)
        Z2, _ = self.attention(Z, Z_g, Z_c1)
        Z3, _ = self.attention(Z, Z_g, Z_c2)
        
        fusion_input = torch.cat([Z1, Z2, Z3], dim=1)
        fusion_weights = self.fusion_gate(fusion_input)
        
        Z4 = (fusion_weights[:, 0:1] * Z1 +
              fusion_weights[:, 1:2] * Z2 +
              fusion_weights[:, 2:3] * Z3)
        
        Z4 = Z4 * self.adaptive_scaling
        Z4 = self.final_projection(Z4)
        
        # Step 4: Calculate alignment loss
        alignment_loss = self.alignment_loss(Z4, x_c)
        
        # Return raw, unweighted losses
        return Z4, Z_c, Z_g, gene_embeddings, alignment_loss, contrast_loss

class SubgraphEncoder(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_vertices, num_subvertices, dropout=0.2):
        super(SubgraphEncoder, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.bottleneck = nn.Linear(hidden_dim, 32)
        
        # ARCHITECTURE UPGRADE: Attention mechanism for pooling
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.bn_bottleneck = nn.BatchNorm1d(32)
        self.bn_g1 = nn.BatchNorm1d(hidden_dim)
        self.bn_g2 = nn.BatchNorm1d(hidden_dim)
        self.dropout_rate = dropout
        self.num_vertices = num_vertices
        self.num_subvertices = num_subvertices
        self.extra_fc = nn.Linear(hidden_dim, hidden_dim)
        self.bn_extra = nn.BatchNorm1d(hidden_dim)
        
    def embed(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn_g1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn_g2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return x
        
    def forward(self, x, edge_index):
        embeddings = self.embed(x, edge_index)

        total_genes = embeddings.shape[0]

        if total_genes % self.num_subvertices != 0:
            raise ValueError(f"The total number of gene nodes ({total_genes}) is not divisible by the number of genes per cell ({self.num_subvertices}).")

        num_cells = total_genes // self.num_subvertices

        # ARCHITECTURE IMPROVEMENT: Replace flatten-and-linear with mean pooling
        # for more efficient and structurally sound feature aggregation.
        
        # Reshape to [num_cells, num_subvertices, hidden_dim] to prepare for pooling
        x_reshaped = embeddings.view(num_cells, self.num_subvertices, embeddings.shape[1])
        
        # ARCHITECTURE UPGRADE: Replace mean pooling with self-attention pooling
        # 1. Calculate attention scores for each gene within a cell
        attention_scores = self.attention_net(x_reshaped)  # Shape: [num_cells, num_subvertices, 1]
        
        # 2. Normalize scores to get weights
        attention_weights = F.softmax(attention_scores, dim=1) # Shape: [num_cells, num_subvertices, 1]
        
        # 3. Apply weights to gene embeddings (weighted sum)
        x = torch.sum(x_reshaped * attention_weights, dim=1) # Shape: [num_cells, hidden_dim]
        # 额外映射层
        x = self.extra_fc(x)
        x = self.bn_extra(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # 应用瓶颈层获得最终特征
        x = self.bottleneck(x)
        x = self.bn_bottleneck(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        return x, embeddings

class GraphEncoder(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, dropout=0.2):
        super(GraphEncoder, self).__init__()
        
        # 多层GCN结构
        self.conv1 = GCNConv(num_features, hidden_dim*2)
        self.conv2 = GCNConv(hidden_dim*2, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)  # 添加第三层
        
        # 批归一化层
        self.bn1 = nn.BatchNorm1d(hidden_dim*2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # 残差连接投影层
        self.residual_projection = nn.Linear(hidden_dim*2, hidden_dim)
        
        # Dropout
        self.dropout_rate = dropout
        
        # 注意力池化层
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=0)
        )

    def forward(self, x, edge_index):
        # 第一层
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout_rate, training=self.training)
        
        # 第二层
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        
        # 残差连接
        residual = self.residual_projection(x1)
        x2 = x2 + residual  # 残差连接
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout_rate, training=self.training)
        
        # 第三层
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = x3 + x2  # 另一个残差连接
        x3 = F.relu(x3)
        
        return x3

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embedding_dim):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)  # apply sin to even indices
        self.encoding[:, 1::2] = torch.cos(position * div_term)  # apply cos to odd indices
        self.encoding = self.encoding.unsqueeze(0)  # add batch dimension
        
    def forward(self, x):
        # 取出当前节点数大小的编码
        return self.encoding[:, :x.size(0), :].to(x.device)
        
class AttentionalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, max_len=5000, dropout=0.2):

        super(AttentionalGNN, self).__init__()
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 位置编码维度应与hidden_dim匹配
        self.pos_encoding_dim = hidden_dim

        self.actual_input_dim = hidden_dim + self.pos_encoding_dim
        
        # 多头注意力层
        self.query_fc = nn.Linear(self.actual_input_dim, hidden_dim * num_heads)
        self.key_fc = nn.Linear(self.actual_input_dim, hidden_dim * num_heads)
        self.value_fc = nn.Linear(self.actual_input_dim, hidden_dim * num_heads)
        
        # 输出投影层
        self.output_fc = nn.Linear(hidden_dim * num_heads, output_dim)
        
        # 残差连接的投影层（维度匹配）
        self.residual_projection = nn.Linear(input_dim, output_dim)
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)
        
        # 前馈网络
        # STRATEGY: Increase depth of FFN to make attention module more powerful.
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.Dropout(dropout)
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 位置编码
        self.position_encoding = PositionalEncoding(max_len, self.pos_encoding_dim)
        
        # 图结构感知权重
        self.structure_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, Z, Z_g, Z_c, adjacency_matrix=None):

        num_nodes = Z.size(0)
        
        # 生成位置编码
        positions = self.position_encoding(Z)  # [1, num_nodes, pos_encoding_dim]
        pos_encoding = positions.squeeze(0)[:num_nodes]  # [num_nodes, pos_encoding_dim]
        
        # 将位置编码添加到特征中
        Z_g_with_pos = torch.cat([Z_g, pos_encoding], dim=-1)
        Z_c_with_pos = torch.cat([Z_c, pos_encoding], dim=-1)
        
        # 计算 Q, K, V
        Q = self.query_fc(Z_c_with_pos)  # [num_nodes, hidden_dim * num_heads]
        K = self.key_fc(Z_g_with_pos)    # [num_nodes, hidden_dim * num_heads]
        V = self.value_fc(Z_g_with_pos)  # [num_nodes, hidden_dim * num_heads]
        
        # 重塑为多头形式
        Q = Q.view(num_nodes, self.num_heads, self.hidden_dim).transpose(0, 1)
        K = K.view(num_nodes, self.num_heads, self.hidden_dim).transpose(0, 1)
        V = V.view(num_nodes, self.num_heads, self.hidden_dim).transpose(0, 1)
        
        # 计算注意力分数
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        
        # 如果提供邻接矩阵，则融入图结构信息
        if adjacency_matrix is not None:
            # 将邻接矩阵扩展到多头
            structure_bias = adjacency_matrix.unsqueeze(0).repeat(self.num_heads, 1, 1)
            # 对非连接的节点添加负偏置
            structure_mask = (structure_bias == 0).float() * (-1e9)
            attention_scores = attention_scores + self.structure_weight * structure_mask
        
        # 应用softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算注意力输出
        attended_values = torch.bmm(attention_weights, V)
        
        # 重塑并投影到输出维度
        attended_values = attended_values.transpose(0, 1).contiguous().view(
            num_nodes, self.num_heads * self.hidden_dim
        )
        
        # 输出投影
        output = self.output_fc(attended_values)
        
        # 残差连接和层归一化
        residual = self.residual_projection(Z)
        output = self.layer_norm1(output + residual)
        
        # 前馈网络 + 残差连接
        ffn_output = self.ffn(output)
        output = self.layer_norm2(output + ffn_output)
        
        return output, attention_weights

class AlignmentLoss(nn.Module):


    def __init__(self, z4_dim, xc_dim, lambda_center=0.99, lambda_structure=0.56,
                 adaptive_weights=False, use_hard_mining=True):
        super(AlignmentLoss, self).__init__()
        if z4_dim <= 0 or xc_dim <= 0:
             raise ValueError("z4_dim and xc_dim must be positive")
             
        self.adaptive_weights = adaptive_weights
        self.use_hard_mining = use_hard_mining
        
        # Use fixed, tunable weights
        if adaptive_weights:
            self.lambda_center = nn.Parameter(torch.tensor(lambda_center))
            self.lambda_structure = nn.Parameter(torch.tensor(lambda_structure))
        else:
            self.lambda_center = lambda_center
            self.lambda_structure = lambda_structure
        
        # 中心投影层 - 使用更复杂的映射
        self.center_projection = nn.Sequential(
            nn.Linear(xc_dim, z4_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(z4_dim * 2, z4_dim)
        )
        
        # 用于困难样本挖掘的阈值
        self.hard_threshold = 0.7

    def _compute_robust_similarity(self, x1, x2, eps=1e-8):

        # L2归一化
        x1_norm = F.normalize(x1, p=2, dim=1, eps=eps)
        x2_norm = F.normalize(x2, p=2, dim=1, eps=eps)
        
        # 计算余弦相似度
        similarity = torch.matmul(x1_norm, x2_norm.T)
        
        # 数值稳定性处理
        similarity = torch.clamp(similarity, -1.0 + eps, 1.0 - eps)
        
        return similarity

    def _hard_sample_mining(self, sim_pred, sim_target):
        if not self.use_hard_mining:
            return torch.ones_like(sim_pred)
            
        # 计算预测与目标的差异
        diff = torch.abs(sim_pred - sim_target)
        
        # 找到困难样本（差异大的样本）
        threshold = torch.quantile(diff, self.hard_threshold)
        hard_mask = (diff >= threshold).float()
        
        # 为困难样本分配更高权重
        weights = torch.ones_like(diff)
        weights = weights + hard_mask  # 困难样本权重加倍
        
        return weights

    def forward(self, Z4, x_c):

        if Z4.shape[0] != x_c.shape[0]:
            raise ValueError("Z4 and x_c must have the same number of nodes (dim 0)")
            
        batch_size = Z4.shape[0]
        
        # --- 改进的结构损失 ---
        # 使用更稳定的相似度计算
        sim_Z4 = self._compute_robust_similarity(Z4, Z4)
        sim_xc = self._compute_robust_similarity(x_c, x_c)
        
        # 困难样本挖掘
        sample_weights = self._hard_sample_mining(sim_Z4, sim_xc)
        
        # 加权结构保持损失
        structure_diff = (sim_Z4 - sim_xc) ** 2
        weighted_structure_diff = structure_diff * sample_weights
        structure_loss = weighted_structure_diff.mean()

        # --- 改进的中心损失 ---
        # 计算自适应中心
        center = torch.mean(x_c, dim=0, keepdim=True)  # [1, xc_dim]
        
        # 更复杂的中心投影
        projected_center = self.center_projection(center)  # [1, z4_dim]
        projected_center = projected_center.to(Z4.device, dtype=Z4.dtype)
        
        # 自适应中心损失 - 考虑样本到中心的距离分布
        center_distances = torch.norm(Z4 - projected_center, p=2, dim=1)
        center_loss = center_distances.mean()
        
        # 添加方差正则化以避免塌陷
        variance_loss = -torch.log(torch.var(Z4, dim=0) + 1e-8).mean()

        # --- 获取当前权重 ---
        if self.adaptive_weights:
            current_lambda_center = torch.clamp(self.lambda_center, 0.01, 10.0)
            current_lambda_structure = torch.clamp(self.lambda_structure, 0.01, 10.0)
        else:
            current_lambda_center = self.lambda_center
            current_lambda_structure = self.lambda_structure

        # --- 总损失 ---
        total_loss = (current_lambda_center * center_loss + 
                     current_lambda_structure * structure_loss +
                     0.1 * variance_loss)  
        
        return total_loss