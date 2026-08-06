"""
INFO 外层超参优化器

基于论文 "INFO-TCN-iTransformer" 公式(3)：
    S(θ) = Mean[R(θ)] - λ * Std[R(θ)]

其中:
- R(θ) 是参数配置 θ 在验证集上的Top-5等权比赛得分
- Mean[R(θ)] 衡量预测精度期望
- Std[R(θ)] 衡量性能波动风险（稳定性惩罚）
- λ 是稳定性惩罚系数

INFO 将模型性能视为随机变量，通过多次重复训练评估其统计特性，
在追求高精度的同时规避高方差参数配置，引导搜索趋向泛化性强的参数区域。
"""

import copy
import itertools
import random
import numpy as np
import torch


# ============================================================
# 超参搜索空间定义
# ============================================================
DEFAULT_SEARCH_SPACE = {
    'learning_rate': [5e-6, 1e-5, 2e-5, 5e-5, 1e-4],
    'tcn_num_layers': [2, 3, 4],
    'tcn_channels': [32, 64, 128],
    'it_num_layers': [1, 2, 3],
    'dropout': [0.05, 0.1, 0.15, 0.2],
}


def sample_config(search_space, base_config, rng=None):
    """从搜索空间中随机采样一组超参配置。

    Parameters
    ----------
    search_space : dict
        超参搜索空间，每个键对应可选值列表。
    base_config : dict
        基础配置字典，采样结果会覆盖其中的搜索参数。
    rng : random.Random, optional
        随机数生成器，用于可重复采样。

    Returns
    -------
    dict : 采样后的配置字典
    """
    if rng is None:
        rng = random.Random()
    cfg = copy.deepcopy(base_config)
    for key, values in search_space.items():
        cfg[key] = rng.choice(values)
    return cfg


def info_score(scores, lambda_stability=0.5):
    """计算 INFO 评分函数（论文公式3）。

    S(θ) = Mean[R(θ)] - λ * Std[R(θ)]

    Parameters
    ----------
    scores : list[float]
        多次重复训练得到的验证集性能分数列表。
    lambda_stability : float
        稳定性惩罚系数 λ。

    Returns
    -------
    float : INFO 评分值（越大越好）
    """
    if len(scores) == 0:
        return -float('inf')
    arr = np.array(scores)
    return float(np.mean(arr) - lambda_stability * np.std(arr))


class INFOOptimizer:
    """INFO 外层超参优化器。

    使用随机搜索策略在预定义的搜索空间中采样参数配置，
    对每组配置进行多次重复训练，用 INFO 评分函数选择最优配置。

    Parameters
    ----------
    base_config : dict
        基础配置字典（包含所有必要参数）。
    search_space : dict, optional
        超参搜索空间，默认使用 DEFAULT_SEARCH_SPACE。
    search_budget : int
        采样的参数配置数量。
    num_trials : int
        每组参数重复训练次数（用于统计稳定性）。
    lambda_stability : float
        稳定性惩罚系数 λ。
    max_epochs : int
        搜索阶段每组参数的训练 epoch 数（快速评估用，应小于正式训练）。
    seed : int
        随机种子。
    """

    def __init__(
        self,
        base_config,
        search_space=None,
        search_budget=10,
        num_trials=3,
        lambda_stability=0.5,
        max_epochs=10,
        seed=42,
    ):
        self.base_config = base_config
        self.search_space = search_space or DEFAULT_SEARCH_SPACE
        self.search_budget = search_budget
        self.num_trials = num_trials
        self.lambda_stability = lambda_stability
        self.max_epochs = max_epochs
        self.seed = seed
        self.rng = random.Random(seed)

        # 记录搜索结果
        self.results = []  # list of (config, scores, info_score_value)
        self.best_config = None
        self.best_info_score = -float('inf')

    def generate_candidates(self):
        """生成 search_budget 个候选参数配置。"""
        candidates = []
        for i in range(self.search_budget):
            cfg = sample_config(self.search_space, self.base_config, self.rng)
            # 搜索阶段使用较少 epoch 加速评估
            cfg['num_epochs'] = self.max_epochs
            # 标记为搜索阶段
            cfg['_info_trial_seed_base'] = self.seed + i * 100
            candidates.append(cfg)
        return candidates

    def evaluate_candidate(self, config, train_fn, eval_fn,
                           train_loader, val_loader, model_builder,
                           device, criterion):
        """评估单个候选配置：多次训练 + 验证，计算 INFO 评分。

        Parameters
        ----------
        config : dict
            候选参数配置。
        train_fn : callable
            训练一个 epoch 的函数: train_fn(model, loader, criterion, optimizer, device, epoch, writer=None)
            返回 (loss, metrics_dict)。
        eval_fn : callable
            评估函数: eval_fn(model, loader, criterion, device, writer=None, epoch=0)
            返回 (loss, metrics_dict)。
        train_loader : DataLoader
        val_loader : DataLoader
        model_builder : callable
            模型构建函数: model_builder(config) -> nn.Module
        device : torch.device
        criterion : nn.Module

        Returns
        -------
        tuple : (config, scores_list, info_score_value)
        """
        scores = []
        seed_base = config.get('_info_trial_seed_base', self.seed)

        for trial in range(self.num_trials):
            trial_seed = seed_base + trial
            # 设置随机种子以保证可重复性
            random.seed(trial_seed)
            np.random.seed(trial_seed)
            torch.manual_seed(trial_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(trial_seed)

            # 构建模型
            model = model_builder(config)
            model.to(device)

            # 构建优化器
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config['learning_rate'],
                weight_decay=1e-5
            )
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.5,
                total_iters=config['num_epochs']
            )

            # 快速训练
            for epoch in range(config['num_epochs']):
                train_fn(model, train_loader, criterion, optimizer, device, epoch, writer=None)
                scheduler.step()

            # 验证
            _, eval_metrics = eval_fn(
                model, val_loader, criterion, device, writer=None, epoch=0
            )
            metric_name = config.get(
                'model_selection_metric', 'top5_return'
            )
            score = eval_metrics.get(metric_name, 0.0)
            scores.append(score)

            # 释放模型显存
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 计算 INFO 评分
        i_score = info_score(scores, self.lambda_stability)

        return config, scores, i_score

    def search(self, train_fn, eval_fn, train_loader, val_loader,
               model_builder, device, criterion, verbose=True):
        """执行 INFO 超参搜索。

        Parameters
        ----------
        train_fn, eval_fn, train_loader, val_loader, model_builder, device, criterion
            同 evaluate_candidate 参数说明。
        verbose : bool
            是否打印搜索进度。

        Returns
        -------
        dict : 最优参数配置
        """
        candidates = self.generate_candidates()

        if verbose:
            print(f"\n{'='*60}")
            print(f"INFO 超参搜索开始")
            print(f"  搜索配置数: {self.search_budget}")
            print(f"  每组重复次数: {self.num_trials}")
            print(f"  搜索阶段 epoch 数: {self.max_epochs}")
            print(f"  稳定性惩罚系数 λ: {self.lambda_stability}")
            print(f"{'='*60}\n")

        for idx, cfg in enumerate(candidates):
            if verbose:
                print(f"[INFO 搜索 {idx+1}/{self.search_budget}] "
                      f"lr={cfg['learning_rate']:.2e}, "
                      f"tcn_layers={cfg['tcn_num_layers']}, "
                      f"tcn_ch={cfg['tcn_channels']}, "
                      f"it_layers={cfg['it_num_layers']}, "
                      f"dropout={cfg['dropout']}")

            cfg_copy, scores, i_score = self.evaluate_candidate(
                cfg, train_fn, eval_fn, train_loader, val_loader,
                model_builder, device, criterion
            )

            self.results.append((cfg_copy, scores, i_score))

            if verbose:
                print(f"  各次得分: {[f'{s:.4f}' for s in scores]}")
                print(f"  Mean={np.mean(scores):.4f}, "
                      f"Std={np.std(scores):.4f}, "
                      f"INFO Score={i_score:.4f}")

            if i_score > self.best_info_score:
                self.best_info_score = i_score
                # 还原为正式训练的 epoch 数
                best_cfg = copy.deepcopy(cfg_copy)
                best_cfg['num_epochs'] = self.base_config['num_epochs']
                best_cfg.pop('_info_trial_seed_base', None)
                self.best_config = best_cfg

                if verbose:
                    print(f"  ★ 当前最优！INFO Score={i_score:.4f}")
            if verbose:
                print()

        if verbose:
            print(f"{'='*60}")
            print(f"INFO 搜索完成！最优 INFO Score: {self.best_info_score:.4f}")
            print(f"最优配置:")
            for k, v in self.best_config.items():
                if not k.startswith('_'):
                    print(f"  {k}: {v}")
            print(f"{'='*60}\n")

        return self.best_config

    def get_search_summary(self):
        """返回搜索结果摘要。"""
        if not self.results:
            return "尚未执行搜索"
        lines = [f"INFO 搜索共评估 {len(self.results)} 组配置"]
        lines.append(f"最优 INFO Score: {self.best_info_score:.4f}")
        lines.append(f"最优配置: {self.best_config}")
        return "\n".join(lines)
