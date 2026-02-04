# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import wandb
from .hook import Hook


class WANDBHook(Hook):
    """
    Wandb Hook
    """

    def __init__(self):
        super().__init__()
        self.log_key_list = ['train/sup_loss', 'train/unsup_loss', 'train/total_loss', 'train/util_ratio', 
                             'train/run_time', 'train/prefetch_time', 'lr',
                             'train/pseudo_acc', 'train/epoch_pseudo_label_count', 'train/epoch_pseudo_label_acc',
                             'eval/pseudo_acc',
                             'eval/top-1-acc', 'eval/precision', 'eval/recall', 'eval/F1']

    def before_run(self, algorithm):
        # job_id = '_'.join(algorithm.args.save_name.split('_')[:-1])
        name = algorithm.save_name
        project = algorithm.save_dir.split('/')[-1]

        # tags
        benchmark = f'benchmark: {project}'
        dataset = f'dataset: {algorithm.args.dataset}'
        data_setting = f'setting: {algorithm.args.dataset}_lb{algorithm.args.num_labels}_{algorithm.args.lb_imb_ratio}_ulb{algorithm.args.ulb_num_labels}_{algorithm.args.ulb_imb_ratio}'
        alg = f'alg: {algorithm.args.algorithm}'
        imb_alg = f'imb_alg: {algorithm.args.imb_algorithm}'
        tags = [benchmark, dataset, data_setting, alg, imb_alg] 
        if algorithm.args.resume:
            resume = 'auto'
        else:
            resume = 'never'
        # resume = 'never'
        
        save_dir = os.path.join(algorithm.args.save_dir, 'wandb', algorithm.args.save_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.run = wandb.init(name=name, 
                              tags=tags, 
                              config=algorithm.args.__dict__, 
                              project=project, 
                              resume=resume,
                              dir=save_dir)


    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_log_iter):
            log_dict = {}
            for key, item in algorithm.log_dict.items():
                if key in self.log_key_list:
                    log_dict[key] = item
            self.run.log(log_dict, step=algorithm.it)
    
        if self.every_n_iters(algorithm, algorithm.num_eval_iter):
            # 记录评估指标，包括伪标签准确率
            eval_log_dict = {'eval/best-acc': algorithm.best_eval_acc}
            for key, item in algorithm.log_dict.items():
                if key.startswith('eval/') and key in self.log_key_list:
                    eval_log_dict[key] = item
            self.run.log(eval_log_dict, step=algorithm.it)
    
    def after_train_epoch(self, algorithm):
        # 记录每轮的伪标签统计信息
        epoch_log_dict = {}
        for key in ['train/epoch_pseudo_label_count', 'train/epoch_pseudo_label_acc']:
            if key in algorithm.log_dict and key in self.log_key_list:
                epoch_log_dict[key] = algorithm.log_dict[key]
        if epoch_log_dict:
            # 使用epoch作为step，或者使用当前的iteration
            self.run.log(epoch_log_dict, step=algorithm.it)
    
    def after_run(self, algorithm):
        self.run.finish()