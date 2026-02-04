# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
from sklearn.cluster import KMeans
from semilearn.algorithms.hooks import MaskingHook


class EMMTThresholdingHook(MaskingHook):
    """
    K-Means Clustering based Thresholding Hook for EMMT.
    使用k-means聚类筛选伪标签，仅使用距离聚类中心较近的无标签样本加入训练。
    聚类结果在训练开始前预计算，每个聚类有独立的动态调整的distance阈值。
    """
    
    def __init__(self, n_clusters=None, distance_ratio=0.5, consistency_threshold=0.5, *args, **kwargs):
        """
        Args:
            n_clusters: k-means聚类的簇数，如果为None则使用类别数
            distance_ratio: 距离比例阈值（已废弃，保留用于兼容性）
            consistency_threshold: 预测一致性阈值，超过此值才赋伪标签
        """
        super().__init__(*args, **kwargs)
        self.n_clusters = n_clusters
        self.distance_ratio = distance_ratio
        self.consistency_threshold = consistency_threshold
        # 预计算的聚类结果
        self.cluster_centers = None
        self.sample_distances = None
        self.cluster_labels = None
        self.sample_indices = None
        self.cluster_distance_thresholds = {}  # {cluster_id: threshold}
        self.use_precomputed = False
    
    def set_clustering_results(self, cluster_centers, sample_distances, cluster_labels, indices, 
                              cluster_distance_thresholds=None):
        """
        设置预计算的聚类结果
        
        Args:
            cluster_centers: 聚类中心 (n_clusters, feature_dim)
            sample_distances: 每个样本到其所属簇中心的距离 (n_samples,)
            cluster_labels: 每个样本所属的簇 (n_samples,)
            indices: 样本的原始索引 (n_samples,)
            cluster_distance_thresholds: 每个聚类的distance阈值 {cluster_id: threshold}
        """
        self.cluster_centers = cluster_centers
        self.sample_distances = sample_distances
        self.cluster_labels = cluster_labels
        self.sample_indices = indices
        if cluster_distance_thresholds is not None:
            self.cluster_distance_thresholds = cluster_distance_thresholds.copy()
        self.use_precomputed = True
    
    def update_cluster_threshold(self, cluster_id, new_threshold):
        """
        更新指定聚类的distance阈值
        
        Args:
            cluster_id: 聚类ID
            new_threshold: 新的阈值
        """
        self.cluster_distance_thresholds[cluster_id] = new_threshold
    
    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, logits_x_ulb_s=None, feats_x_ulb_w=None, idx_ulb=None, softmax_x_ulb=True, *args, **kwargs):
        """
        使用预计算的k-means聚类结果和动态阈值筛选伪标签
        
        Args:
            algorithm: 算法实例
            logits_x_ulb: 无标签数据的logits（弱增强）
            logits_x_ulb_s: 无标签数据的logits（强增强），用于计算预测一致性
            feats_x_ulb_w: 无标签数据的特征向量（已废弃）
            idx_ulb: 无标签数据的索引，用于查找预计算的聚类结果
            softmax_x_ulb: 是否对logits进行softmax
        
        Returns:
            mask: 伪标签mask
            cluster_info: 每个聚类的预测信息 {cluster_id: [predictions]}，用于后续调整阈值
        """
        if softmax_x_ulb:
            probs_x_ulb = algorithm.compute_prob(logits_x_ulb.detach())
        else:
            probs_x_ulb = logits_x_ulb.detach()
        
        # 首先使用置信度阈值进行初步筛选
        max_probs, _ = torch.max(probs_x_ulb, dim=-1)
        confidence_mask = max_probs.ge(algorithm.p_cutoff)
        
        # 如果没有样本通过置信度阈值，返回全0的mask
        if confidence_mask.sum() == 0:
            return torch.zeros_like(max_probs, dtype=max_probs.dtype, device=max_probs.device), None
        
        # 如果使用预计算的聚类结果
        if self.use_precomputed and self.sample_distances is not None and idx_ulb is not None:
            try:
                # 获取当前batch中样本的索引
                batch_indices = idx_ulb.cpu().numpy()
                
                # 找到这些索引在预计算结果中的位置
                index_map = {idx: i for i, idx in enumerate(self.sample_indices)}
                
                # 获取当前batch中通过置信度阈值的样本
                confident_indices = torch.where(confidence_mask)[0].cpu().numpy()
                batch_confident_indices = batch_indices[confident_indices]
                
                # 获取弱增强和强增强的预测
                pred_ulb_w = torch.argmax(logits_x_ulb, dim=-1).cpu().numpy()
                if logits_x_ulb_s is not None:
                    pred_ulb_s = torch.argmax(logits_x_ulb_s, dim=-1).cpu().numpy()
                else:
                    pred_ulb_s = pred_ulb_w.copy()
                
                # 按聚类组织样本
                cluster_samples = {}  # {cluster_id: [(batch_idx_in_confident, orig_idx, distance, pred_w, pred_s)]}
                
                for i, orig_idx in enumerate(batch_confident_indices):
                    if orig_idx in index_map:
                        sample_idx = index_map[orig_idx]
                        cluster_id = self.cluster_labels[sample_idx]
                        distance = self.sample_distances[sample_idx]
                        batch_idx = confident_indices[i]  # 在原始batch中的索引
                        
                        if cluster_id not in cluster_samples:
                            cluster_samples[cluster_id] = []
                        
                        cluster_samples[cluster_id].append((
                            batch_idx,
                            orig_idx,
                            distance,
                            pred_ulb_w[batch_idx],
                            pred_ulb_s[batch_idx]
                        ))
                
                # 对每个聚类应用distance阈值并筛选
                final_mask = torch.zeros_like(max_probs, dtype=max_probs.dtype, device=max_probs.device)
                cluster_info = {}  # 用于跟踪每个聚类的预测
                
                for cluster_id, samples in cluster_samples.items():
                    if cluster_id not in self.cluster_distance_thresholds:
                        continue
                    
                    threshold = self.cluster_distance_thresholds[cluster_id]
                    
                    # 根据distance阈值筛选样本
                    filtered_samples = [s for s in samples if s[2] <= threshold]
                    
                    if len(filtered_samples) == 0:
                        continue
                    
                    # 获取筛选后样本的预测（使用强增强的预测）
                    predictions = np.array([s[4] for s in filtered_samples])
                    
                    # 计算预测一致性
                    unique, counts = np.unique(predictions, return_counts=True)
                    max_count = np.max(counts) if len(counts) > 0 else 0
                    consistency_ratio = max_count / len(predictions) if len(predictions) > 0 else 0
                    
                    # 记录预测信息用于后续调整阈值
                    cluster_info[cluster_id] = predictions.tolist()
                    
                    # 如果超过一致性阈值，将预测最多的样本赋伪标签
                    if consistency_ratio >= self.consistency_threshold:
                        most_common_pred = unique[np.argmax(counts)]
                        
                        # 只选择预测为most_common_pred的样本
                        for batch_idx, orig_idx, distance, pred_w, pred_s in filtered_samples:
                            if pred_s == most_common_pred:
                                final_mask[batch_idx] = 1.0
                
                return final_mask, cluster_info
                    
            except Exception as e:
                # 如果使用预计算结果失败，回退到置信度阈值
                if hasattr(algorithm, 'print_fn'):
                    algorithm.print_fn(f"Warning: Failed to use precomputed clustering: {e}, falling back to confidence thresholding")
                mask = confidence_mask.to(max_probs.dtype)
                return mask, None
        
        # 如果没有预计算结果，回退到置信度阈值
        if hasattr(algorithm, 'print_fn'):
            algorithm.print_fn("Warning: No precomputed clustering results, using confidence threshold only")
        mask = confidence_mask.to(max_probs.dtype)
        return mask, None

