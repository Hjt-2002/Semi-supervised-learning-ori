# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.algorithms.emmt.utils import EMMTThresholdingHook

# 设置multiprocessing sharing strategy以避免"Too many open files"错误
try:
    torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
    # 如果已经设置过，忽略错误
    pass


@ALGORITHMS.register('emmt')
class EMMT(AlgorithmBase):

    """
        EMMT algorithm - 基于FixMatch的改进版本
        在筛选伪标签时运用k-means聚类，仅使用距离聚类中心较近的无标签样本加入训练。

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
            - n_clusters (`int`, *optional*, default to `None`):
                k-means聚类的簇数，如果为None则使用类别数
            - distance_ratio (`float`, *optional*, default to `0.5`):
                距离比例阈值，只选择距离聚类中心距离小于该比例的样本
    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        # fixmatch specified arguments
        n_clusters = getattr(args, 'n_clusters', None)
        distance_ratio = getattr(args, 'distance_ratio', 0.5)
        init_distance_percentile = getattr(args, 'init_distance_percentile', 0.5)
        consistency_threshold = getattr(args, 'consistency_threshold', 0.5)
        consistency_threshold1 = getattr(args, 'consistency_threshold1', None)
        consistency_threshold2 = getattr(args, 'consistency_threshold2', None)
        threshold_decay_low = getattr(args, 'threshold_decay_low', 0.7)
        threshold_decay_mid = getattr(args, 'threshold_decay_mid', 0.9)
        threshold_grow = getattr(args, 'threshold_grow', 1.1)
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label,
                  n_clusters=n_clusters, distance_ratio=distance_ratio,
                  init_distance_percentile=init_distance_percentile,
                  consistency_threshold=consistency_threshold,
                  consistency_threshold1=consistency_threshold1,
                  consistency_threshold2=consistency_threshold2,
                  threshold_decay_low=threshold_decay_low,
                  threshold_decay_mid=threshold_decay_mid,
                  threshold_grow=threshold_grow)
    
    def init(self, T, p_cutoff, hard_label=True, n_clusters=None, distance_ratio=0.5,
             init_distance_percentile=0.5, consistency_threshold=0.5,
             consistency_threshold1=None, consistency_threshold2=None,
             threshold_decay_low=0.7, threshold_decay_mid=0.9, threshold_grow=1.1):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.n_clusters = n_clusters
        self.distance_ratio = distance_ratio
        # 动态阈值调整超参数
        self.init_distance_percentile = init_distance_percentile  # 初始distance阈值百分位数
        self.consistency_threshold = consistency_threshold  # 伪标签一致性阈值（用于masking）
        if consistency_threshold1 is None:
            consistency_threshold1 = consistency_threshold
        if consistency_threshold2 is None:
            consistency_threshold2 = 0.9
        self.consistency_threshold1 = consistency_threshold1  # 低一致性阈值
        self.consistency_threshold2 = consistency_threshold2  # 高一致性阈值
        self.threshold_decay_low = threshold_decay_low  # 低一致性时的衰减因子（0.7）
        self.threshold_decay_mid = threshold_decay_mid  # 中等一致性时的衰减因子（0.9）
        self.threshold_grow = threshold_grow  # 高一致性时的增长因子（1.1）
        # 聚类结果存储
        self.cluster_centers = None
        self.sample_distances = None  # 每个样本到其所属簇中心的距离
        self.cluster_labels = None  # 每个样本所属的簇
        self.cluster_distance_thresholds = None  # 每个聚类的distance阈值
        self.clustering_initialized = False
        # 用于跟踪每个聚类的预测一致性
        self.cluster_predictions = {}  # {cluster_id: [predictions]}
        # 用于跟踪每轮的伪标签统计
        self.epoch_pseudo_label_count = 0  # 每轮使用的伪标签数量
        self.epoch_pseudo_label_acc_sum = 0.0  # 每轮伪标签准确率总和
        self.epoch_pseudo_label_batch_count = 0  # 每轮有伪标签的batch数量
    
    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        # 使用K-Means聚类阈值hook替代固定阈值hook
        # 从args获取参数，因为此时init方法可能还未被调用
        # self.args在super().__init__中已经被设置，可以直接使用
        n_clusters = getattr(self.args, 'n_clusters', None)
        distance_ratio = getattr(self.args, 'distance_ratio', 0.5)
        consistency_threshold = getattr(self.args, 'consistency_threshold', 0.5)
        
        # 如果init已经被调用，优先使用self中的值（更准确）
        if hasattr(self, 'n_clusters'):
            n_clusters = self.n_clusters
        if hasattr(self, 'distance_ratio'):
            distance_ratio = self.distance_ratio
        if hasattr(self, 'consistency_threshold'):
            consistency_threshold = self.consistency_threshold
        
        self.register_hook(
            EMMTThresholdingHook(
                n_clusters=n_clusters,
                distance_ratio=distance_ratio,
                consistency_threshold=consistency_threshold
            ),
            "MaskingHook"
        )
        super().set_hooks()
    
    def initialize_clustering(self):
        """
        在训练开始前，使用预训练模型对所有无标签数据进行聚类
        提取特征 -> k-means聚类 -> 计算距离 -> 保存结果
        """
        if self.clustering_initialized:
            return
        
        if self.dataset_dict is None or self.dataset_dict['train_ulb'] is None:
            self.print_fn("Warning: No unlabeled data available for clustering")
            return
        
        self.print_fn("Initializing clustering with pretrained model...")
        
        # 创建用于提取特征的dataloader（不使用数据增强，只使用弱增强）
        # 使用较少的workers避免"Too many open files"错误
        ulb_dataset = self.dataset_dict['train_ulb']
        # 对于特征提取，使用较少的workers或0（单进程）
        num_workers_for_clustering = min(2, self.args.num_workers) if self.args.num_workers > 0 else 0
        feature_loader = DataLoader(
            ulb_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=num_workers_for_clustering,
            drop_last=False,
            pin_memory=False  # 特征提取不需要pin_memory
        )
        
        # 提取所有无标签数据的特征
        self.model.eval()
        all_features = []
        all_indices = []
        
        with torch.no_grad():
            for data_dict in feature_loader:
                idx_ulb = data_dict['idx_ulb']
                x_ulb_w = data_dict['x_ulb_w'].cuda(self.gpu)
                
                # 使用预训练模型提取特征
                outputs = self.model(x_ulb_w)
                features = outputs['feat'].detach().cpu().numpy()
                
                all_features.append(features)
                all_indices.append(idx_ulb.numpy())
        
        # 合并所有特征
        all_features = np.concatenate(all_features, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)
        
        self.print_fn(f"Extracted features for {len(all_features)} unlabeled samples")
        
        # 确定聚类数量
        n_clusters = self.n_clusters
        if n_clusters is None:
            n_clusters = self.num_classes
        n_clusters = min(n_clusters, len(all_features))
        
        if n_clusters < 2:
            self.print_fn("Warning: Too few samples for clustering, using all samples")
            self.clustering_initialized = True
            return
        
        # 执行k-means聚类
        try:
            self.print_fn(f"Performing k-means clustering with {n_clusters} clusters...")
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
            cluster_labels = kmeans.fit_predict(all_features)
            cluster_centers = kmeans.cluster_centers_
            
            # 计算每个样本到其所属聚类中心的距离
            sample_distances = []
            for i, feature in enumerate(all_features):
                cluster_id = cluster_labels[i]
                center = cluster_centers[cluster_id]
                distance = np.linalg.norm(feature - center)
                sample_distances.append(distance)
            
            sample_distances = np.array(sample_distances)
            
            # 为每个聚类计算初始distance阈值（使用50%分位数）
            cluster_distance_thresholds = {}
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_dists = sample_distances[cluster_mask]
                if len(cluster_dists) > 0:
                    # 使用指定的百分位数作为初始阈值
                    threshold = np.percentile(cluster_dists, self.init_distance_percentile * 100)
                    cluster_distance_thresholds[cluster_id] = threshold
                else:
                    cluster_distance_thresholds[cluster_id] = np.mean(sample_distances)
            
            self.print_fn(f"Initialized distance thresholds for {len(cluster_distance_thresholds)} clusters")
            for cluster_id, threshold in cluster_distance_thresholds.items():
                self.print_fn(f"  Cluster {cluster_id}: threshold = {threshold:.4f}")
            
            # 保存聚类结果
            self.cluster_centers = cluster_centers
            self.sample_distances = sample_distances
            self.cluster_labels = cluster_labels
            self.cluster_distance_thresholds = cluster_distance_thresholds
            
            # 保存到文件（可选）
            save_path = os.path.join(self.save_dir, self.save_name)
            os.makedirs(save_path, exist_ok=True)
            clustering_file = os.path.join(save_path, 'clustering_results.npz')
            np.savez(
                clustering_file,
                cluster_centers=cluster_centers,
                sample_distances=sample_distances,
                cluster_labels=cluster_labels,
                indices=all_indices
            )
            self.print_fn(f"Clustering results saved to {clustering_file}")
            
            # 将聚类结果传递给hook
            masking_hook = self.hooks_dict.get("MaskingHook")
            if masking_hook is not None:
                masking_hook.set_clustering_results(
                    cluster_centers=cluster_centers,
                    sample_distances=sample_distances,
                    cluster_labels=cluster_labels,
                    indices=all_indices,
                    cluster_distance_thresholds=cluster_distance_thresholds
                )
            
            self.clustering_initialized = True
            self.print_fn("Clustering initialization completed!")
            
        except Exception as e:
            self.print_fn(f"Error during clustering: {e}")
            self.clustering_initialized = False
    
    def train(self):
        """
        重写train方法，在训练开始前初始化聚类，并在每轮结束后调整阈值
        """
        # 在训练开始前初始化聚类
        if not self.clustering_initialized:
            self.initialize_clustering()
        
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            
            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break
            
            # 每轮开始时清空预测记录和伪标签统计
            self.cluster_predictions = {}
            self.epoch_pseudo_label_count = 0
            self.epoch_pseudo_label_acc_sum = 0.0
            self.epoch_pseudo_label_batch_count = 0
            
            self.call_hook("before_train_epoch")

            for data_lb, data_ulb in zip(self.loader_dict['train_lb'],
                                         self.loader_dict['train_ulb']):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.out_dict, self.log_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1
            
            # 每轮结束后调整distance阈值
            self.adjust_cluster_thresholds()
            
            # 计算并记录每轮的伪标签统计信息
            if self.epoch_pseudo_label_batch_count > 0:
                avg_pseudo_acc = self.epoch_pseudo_label_acc_sum / self.epoch_pseudo_label_batch_count
            else:
                avg_pseudo_acc = 0.0
            
            # 将伪标签统计信息添加到log_dict，供wandb记录
            self.log_dict['train/epoch_pseudo_label_count'] = self.epoch_pseudo_label_count
            self.log_dict['train/epoch_pseudo_label_acc'] = avg_pseudo_acc
            
            self.call_hook("after_train_epoch")

        self.call_hook("after_run")
    
    def adjust_cluster_thresholds(self):
        """
        根据每个聚类的预测一致性调整distance阈值
        """
        if self.cluster_distance_thresholds is None:
            return
        
        masking_hook = self.hooks_dict.get("MaskingHook")
        if masking_hook is None:
            return
        
        for cluster_id in self.cluster_distance_thresholds.keys():
            if cluster_id not in self.cluster_predictions or len(self.cluster_predictions[cluster_id]) == 0:
                continue
            
            predictions = np.array(self.cluster_predictions[cluster_id])
            
            # 计算预测一致性（最常见的预测占比）
            unique, counts = np.unique(predictions, return_counts=True)
            max_count = np.max(counts) if len(counts) > 0 else 0
            consistency_ratio = max_count / len(predictions) if len(predictions) > 0 else 0
            
            # 根据一致性调整阈值
            current_threshold = self.cluster_distance_thresholds[cluster_id]
            
            if consistency_ratio < self.consistency_threshold1:
                # 少于consistency_threshold1一致：阈值 × threshold_decay_low
                new_threshold = current_threshold * self.threshold_decay_low
            elif consistency_ratio > self.consistency_threshold2:
                # 多于consistency_threshold2一致：阈值 × threshold_grow
                new_threshold = current_threshold * self.threshold_grow
            else:
                # 中等一致性：阈值 × threshold_decay_mid
                new_threshold = current_threshold * self.threshold_decay_mid
            
            self.cluster_distance_thresholds[cluster_id] = new_threshold
            
            # 更新hook中的阈值
            masking_hook.update_cluster_threshold(cluster_id, new_threshold)
        
        # 清空预测记录
        self.cluster_predictions = {}

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, idx_ulb=None, y_ulb=None):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            
            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())
            
            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

            # compute mask using precomputed k-means clustering with dynamic thresholds
            # 传入索引和预测结果用于查找预计算的聚类结果和跟踪一致性
            masking_result = self.call_hook("masking", "MaskingHook", 
                                            logits_x_ulb=probs_x_ulb_w,
                                            logits_x_ulb_s=logits_x_ulb_s,
                                            idx_ulb=idx_ulb,
                                            softmax_x_ulb=False)
            
            # 处理masking返回值（可能是单个mask或(mask, cluster_info)元组）
            if isinstance(masking_result, tuple):
                mask, cluster_info = masking_result
            else:
                mask = masking_result
                cluster_info = None
            
            # 收集每个聚类的预测信息用于后续调整阈值
            if cluster_info is not None:
                for cluster_id, predictions in cluster_info.items():
                    if cluster_id not in self.cluster_predictions:
                        self.cluster_predictions[cluster_id] = []
                    self.cluster_predictions[cluster_id].extend(predictions)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label,
                                               'ce',
                                               mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss

            # 计算伪标签准确率（伪标签与真实标签的一致性）
            # 对于被mask选中的样本，计算伪标签与真实标签的一致性
            with torch.no_grad():
                # 获取伪标签的硬标签形式
                if isinstance(pseudo_label, torch.Tensor):
                    if pseudo_label.dim() == 1:
                        pseudo_label_hard = pseudo_label
                    else:
                        pseudo_label_hard = torch.argmax(pseudo_label, dim=-1)
                else:
                    pseudo_label_hard = torch.argmax(probs_x_ulb_w, dim=-1)
                
                # 从数据集获取真实标签（如果可用）
                y_ulb_true = None
                if idx_ulb is not None and self.dataset_dict is not None and self.dataset_dict['train_ulb'] is not None:
                    ulb_dataset = self.dataset_dict['train_ulb']
                    if hasattr(ulb_dataset, 'targets') and ulb_dataset.targets is not None:
                        # 从数据集获取真实标签
                        batch_indices = idx_ulb.cpu().numpy()
                        y_ulb_true = torch.tensor([ulb_dataset.targets[idx] for idx in batch_indices], 
                                                  dtype=torch.long, device=mask.device)
                
                # 如果有真实标签，计算伪标签与真实标签的一致性
                if y_ulb_true is not None and mask.sum() > 0:
                    # 计算伪标签与真实标签的一致性
                    pseudo_acc = (pseudo_label_hard == y_ulb_true).float()
                    # 只计算被mask选中的样本的准确率
                    pseudo_acc_masked = (pseudo_acc * mask).sum() / mask.sum()
                    # 累计每轮的伪标签统计
                    self.epoch_pseudo_label_count += mask.sum().item()
                    self.epoch_pseudo_label_acc_sum += pseudo_acc_masked.item()
                    self.epoch_pseudo_label_batch_count += 1
                else:
                    # 如果没有真实标签，计算伪标签与强增强预测的一致性（作为替代）
                    pred_ulb_s = torch.argmax(logits_x_ulb_s, dim=-1)
                    pseudo_acc = (pseudo_label_hard == pred_ulb_s).float()
                    if mask.sum() > 0:
                        pseudo_acc_masked = (pseudo_acc * mask).sum() / mask.sum()
                        # 累计每轮的伪标签统计
                        self.epoch_pseudo_label_count += mask.sum().item()
                        self.epoch_pseudo_label_acc_sum += pseudo_acc_masked.item()
                        self.epoch_pseudo_label_batch_count += 1
                    else:
                        pseudo_acc_masked = torch.tensor(0.0, device=mask.device)

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item(),
                                         pseudo_acc=pseudo_acc_masked.item())
        return out_dict, log_dict
    

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--n_clusters', int, None),
            SSL_Argument('--distance_ratio', float, 0.5),
            SSL_Argument('--init_distance_percentile', float, 0.5),
            SSL_Argument('--consistency_threshold', float, 0.5),
            SSL_Argument('--consistency_threshold1', float, 0.5),
            SSL_Argument('--consistency_threshold2', float, 0.9),
            SSL_Argument('--threshold_decay_low', float, 0.7),
            SSL_Argument('--threshold_decay_mid', float, 0.9),
            SSL_Argument('--threshold_grow', float, 1.1),
        ]

