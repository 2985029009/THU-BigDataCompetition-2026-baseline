import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import config
from model import INFO_TCN_iTransformer, AuxiliaryRegressionLoss
from utils import (
    BASELINE24_FEATURES,
    COMPONENT2_FEATURES,
    CROSS4_FEATURES,
    MARKET4_FEATURES,
    add_causal_market_features,
    add_market_component_cross_features,
    add_market_cross_features,
    engineer_features_baseline24,
    fit_market_stress_statistics,
    returns_to_relevance,
)
from utils import create_ranking_dataset_vectorized
from info_optimizer import INFOOptimizer
from pit_data import (
    apply_contest_score_label,
    point_in_time_cross_sectional_rank,
    sha256_file,
)
from stratified_sampling import (
    AuditedStratifiedDateSampler,
    apply_market_state_labels,
    build_causal_market_state_table,
    fit_market_state_thresholds,
)
import joblib
import os
import json
import multiprocessing as mp
import random
import argparse
import shutil
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def build_linear_scheduler(optimizer, cfg):
    """构造与实际停止轮数解耦的固定总长度学习率计划。"""
    total_epochs = int(cfg.get('scheduler_total_epochs', cfg['num_epochs']))
    if total_epochs <= 0:
        raise ValueError("scheduler_total_epochs必须为正数")
    return torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.2,
        total_iters=total_epochs,
    )


def clip_gradients(parameters, max_grad_norm):
    """检查梯度有限性并按显式阈值裁剪，返回可记录的审计统计。"""
    parameters = [
        parameter for parameter in parameters if parameter.grad is not None
    ]
    if not parameters:
        return {
            'grad_norm_before_clip': 0.0,
            'max_grad_norm': max_grad_norm,
            'gradient_clipped': False,
        }
    squared_norm = torch.stack(
        [parameter.grad.detach().float().norm(2).square() for parameter in parameters]
    ).sum()
    grad_norm = squared_norm.sqrt()
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("检测到非有限梯度，已中止参数更新")

    enabled = max_grad_norm is not None and float(max_grad_norm) > 0
    if enabled:
        torch.nn.utils.clip_grad_norm_(
            parameters, float(max_grad_norm), error_if_nonfinite=True
        )
    return {
        'grad_norm_before_clip': float(grad_norm.item()),
        'max_grad_norm': max_grad_norm,
        'gradient_clipped': bool(enabled and grad_norm.item() > float(max_grad_norm)),
    }


def summarize_daily_ics(daily_ics):
    """汇总逐日IC。"""
    values = np.asarray(daily_ics, dtype=np.float64)
    if values.size == 0:
        return {
            'mean_ic': 0.0,
            'ic_std': 0.0,
            'icir': 0.0,
        }
    mean_ic = float(values.mean())
    std_ic = float(values.std())
    return {
        'mean_ic': mean_ic,
        'ic_std': std_ic,
        'icir': mean_ic / std_ic if std_ic > 0 else 0.0,
    }


def summarize_target_distributions(relevance_scores, criterion):
    """汇总各横截面的Listwise目标集中度，供运行快照审计。"""
    ranker = criterion.ranknet if hasattr(criterion, 'ranknet') else criterion
    if not hasattr(ranker, 'target_distribution'):
        sizes = [int(torch.as_tensor(values).numel()) for values in relevance_scores]
        return {
            'target_type': 'top_k_membership',
            'head_k': int(getattr(ranker, 'k', 5)),
            'num_cross_sections': len(sizes),
            'num_stocks': {
                'min': min(sizes) if sizes else None,
                'mean': float(np.mean(sizes)) if sizes else None,
                'max': max(sizes) if sizes else None,
            },
        }
    rows = []
    with torch.no_grad():
        for relevance in relevance_scores:
            values = torch.as_tensor(relevance, dtype=torch.float32).unsqueeze(0)
            probs = ranker.target_distribution(values).squeeze(0)
            sorted_probs = probs.sort(descending=True).values
            row = {
                'num_stocks': int(probs.numel()),
                'top1_mass': float(sorted_probs[:1].sum()),
                'top5_mass': float(sorted_probs[:5].sum()),
                'top10_mass': float(sorted_probs[:10].sum()),
                'entropy': float(
                    -(probs * probs.clamp_min(1e-30).log()).sum()
                ),
            }
            rows.append(row)
    summary = {
        'target_type': ranker.target_type,
        'temperature': ranker.temperature,
        'num_cross_sections': len(rows),
    }
    for key in ('num_stocks', 'top1_mass', 'top5_mass', 'top10_mass', 'entropy'):
        values = [row[key] for row in rows]
        summary[key] = {
            'min': min(values) if values else None,
            'mean': float(np.mean(values)) if values else None,
            'max': max(values) if values else None,
        }
    return summary


feature_cloums_map = {
    'baseline24': ['instrument'] + BASELINE24_FEATURES,
    'baseline24_market4': ['instrument'] + BASELINE24_FEATURES + MARKET4_FEATURES,
    'baseline24_market4_cross4': (
        ['instrument'] + BASELINE24_FEATURES + MARKET4_FEATURES + CROSS4_FEATURES
    ),
    'baseline24_market4_component2': (
        ['instrument'] + BASELINE24_FEATURES + MARKET4_FEATURES + COMPONENT2_FEATURES
    ),
}
feature_engineer_func_map = {
    'baseline24': engineer_features_baseline24,
    'baseline24_market4': engineer_features_baseline24,
    'baseline24_market4_cross4': engineer_features_baseline24,
    'baseline24_market4_component2': engineer_features_baseline24,
}


def _build_label_and_clean(processed, drop_small_open=True):
    """统一构建标签并清洗无效样本。"""
    if 'label' in processed.columns:
        processed['label'] = pd.to_numeric(processed['label'], errors='coerce')
        return processed

    processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
    processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)

    # 过滤无效开盘价，避免收益率极端爆炸
    if drop_small_open:
        processed = processed[processed['open_t1'] > 1e-4]

    processed['label'] = (processed['open_t5'] - processed['open_t1']) / (processed['open_t1'] + 1e-12)
    processed.drop(columns=['open_t1', 'open_t5'], inplace=True)
    return processed


def _preprocess_common(df, stockid2idx, desc, drop_small_open=True):
    assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
    assert stockid2idx is not None, "stockid2idx 不能为空"
    feature_engineer = feature_engineer_func_map[config['feature_num']]
    feature_columns = feature_cloums_map[config['feature_num']]

    # 保证时序正确，避免 shift 标签错位
    df = df.copy()
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    if config.get('label_mode') == 'contest_score':
        df = apply_contest_score_label(df)
    if config['feature_num'] in {
        'baseline24_market4',
        'baseline24_market4_cross4',
        'baseline24_market4_component2',
    }:
        df = add_causal_market_features(df)
    audit_columns = [
        column for column in [
            'label', 'is_member', 'is_suspended', 'is_tradable',
            'in_signal_range', 'label_entry_failed', 'adj_factor', 'source',
        ] if column in df.columns
    ]
    audit_frame = df[['股票代码', '日期'] + audit_columns].copy()

    print(f"正在使用多进程进行{desc}...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    if len(groups) == 0:
        raise ValueError(f"{desc}输入为空，无法继续")

    num_processes = min(10, mp.cpu_count())
    with mp.Pool(processes=num_processes) as pool:
        processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc=desc))

    processed = pd.concat(processed_list).reset_index(drop=True)
    if audit_columns:
        processed = processed.drop(columns=audit_columns, errors='ignore').merge(
            audit_frame,
            on=['股票代码', '日期'],
            how='left',
            validate='one_to_one',
        )

    # 映射股票索引，并剔除映射失败样本
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    processed = _build_label_and_clean(processed, drop_small_open=drop_small_open)
    return processed, feature_columns


# 数据预处理函数
def preprocess_data(df, is_train=True, stockid2idx=None):
    if not is_train:
        return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=False)
    return _preprocess_common(df, stockid2idx, desc="特征工程", drop_small_open=True)


def preprocess_val_data(df, stockid2idx=None):
    # 验证集与训练集保持同口径，避免 label 分布漂移
    return _preprocess_common(df, stockid2idx, desc="验证集特征工程", drop_small_open=True)


# 加权的排序损失函数
class WeightedRankingLoss(nn.Module):
    """
    组合的加权排序损失函数，着重强调top-k的样本。
    """
    def __init__(
        self, temperature=1.0, k=5, weight_factor=2.0,
        pairwise_weight=1, base_weight=1.0,
        target_type='normalized_rank_softmax',
    ):
        super(WeightedRankingLoss, self).__init__()
        self.temperature = temperature
        self.k = k
        self.weight_factor = weight_factor
        self.pairwise_weight = pairwise_weight
        self.base_weight = base_weight
        self.target_type = target_type

    def target_distribution(self, relevance):
        """构造可审计的Listwise目标概率分布。"""
        relevance = relevance.float()
        if self.target_type == 'ordinal_softmax':
            logits = relevance
            return F.softmax(logits / self.temperature, dim=1)
        rel_min = relevance.min(dim=1, keepdim=True).values
        rel_max = relevance.max(dim=1, keepdim=True).values
        normalized = (relevance - rel_min) / (rel_max - rel_min).clamp_min(1e-12)
        if self.target_type == 'normalized_rank_softmax':
            return F.softmax(normalized / self.temperature, dim=1)
        if self.target_type == 'percentile_target':
            # 平均秩缩放到[0,1]后直接作为非负质量；全并列时退化为均匀分布。
            mass = normalized
            zero_mass = mass.sum(dim=1, keepdim=True) <= 1e-12
            mass = torch.where(zero_mass, torch.ones_like(mass), mass)
            return mass / mass.sum(dim=1, keepdim=True)
        raise ValueError(f"未知 listwise_target_type: {self.target_type}")

    def listwise_loss(self, y_pred, y_true, weights):
        """加权的Listwise损失 (KL散度 + Cross Entropy)"""
        
        target_probs = self.target_distribution(y_true)

        # 先将top-k权重作用到目标分布，再重新归一化为概率分布。
        # target_probs本身已归一化，不能再用约N个股票的weights.sum()缩小损失；
        # 旧写法会让Listwise相对Pairwise小约两个数量级。
        weighted_target_probs = target_probs * weights
        weighted_target_probs = weighted_target_probs / (
            weighted_target_probs.sum(dim=1, keepdim=True) + 1e-12
        )
        ce_loss = -(
            weighted_target_probs * F.log_softmax(
                y_pred / self.temperature, dim=1
            )
        ).sum(dim=1).mean()
        
        return ce_loss

    def pairwise_loss(self, y_pred, y_true, weights):
        """加权的Pairwise损失"""
        batch_size, num_items = y_pred.size()
        
        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)
        
        # 只考虑真实标签不同的项目对
        mask = (true_diff != 0).float()
        
        # 创建权重矩阵
        # 如果一对(i, j)中，i或j是关键样本，则权重更高
        weight_matrix = weights.unsqueeze(2) + weights.unsqueeze(1)
        # weight_matrix = torch.where(weight_matrix > 2.0, self.weight_factor, 1.0)
        
        # 标准 RankNet logistic loss。相比有界 sigmoid 损失，
        # softplus 会对高置信度错序保留有效梯度。
        signed_margin = pred_diff * torch.sign(true_diff)
        pairwise_loss = F.softplus(-signed_margin)
        
        # 应用mask和权重
        weighted_loss = pairwise_loss * mask * weight_matrix
        
        num_pairs = mask.sum(dim=[1, 2]).clamp(min=1)
        loss = (weighted_loss.sum(dim=[1, 2]) / num_pairs).mean()
        
        return loss
        
    def forward_with_components(self, y_pred, y_true):
        """
        y_pred: [batch, num_items]
        y_true: [batch, num_items] (真实涨跌幅)
        """
        batch_size, num_items = y_true.size()
        k = min(self.k, num_items)

        # 1. 识别 top-k 的样本
        top_threshold = torch.topk(y_true, k, dim=1).values[:, -1:]

        # 2. 创建权重向量；边界并列项获得相同权重，避免行顺序破坏语义。
        weights = torch.full_like(y_true, fill_value=self.base_weight)
        weights = torch.where(
            y_true >= top_threshold,
            torch.as_tensor(self.weight_factor, device=y_true.device),
            weights,
        )
            
        # 3. 计算加权损失
        listwise = self.listwise_loss(y_pred, y_true, weights)
        pairwise = self.pairwise_loss(y_pred, y_true, weights)
        self.last_components = {
            'listwise_loss_raw': float(listwise.detach()),
            'pairwise_loss_raw': float(pairwise.detach()),
        }
        components = {
            'listwise': listwise,
            'pairwise': self.pairwise_weight * pairwise,
        }
        return sum(components.values()), components

    def forward(self, y_pred, y_true):
        total_loss, _ = self.forward_with_components(y_pred, y_true)
        return total_loss


class RankNetStabilityLoss(nn.Module):
    """修正RankNet + 负相关日下行惩罚；正相关截面不施加稳定性惩罚。"""

    def __init__(self, cfg, eps=1e-8):
        super().__init__()
        self.ranknet = WeightedRankingLoss(
            k=5,
            temperature=cfg.get('listwise_temperature', 1.0),
            target_type=cfg.get(
                'listwise_target_type', 'normalized_rank_softmax'
            ),
            weight_factor=cfg['top5_weight'],
            pairwise_weight=cfg['pairwise_weight'],
            base_weight=cfg.get('base_weight', 1.0),
        )
        self.eps = eps

    def forward_with_components(self, y_pred, y_true):
        pred_centered = y_pred - y_pred.mean(dim=1, keepdim=True)
        true_float = y_true.float()
        true_centered = true_float - true_float.mean(dim=1, keepdim=True)
        numerator = (pred_centered * true_centered).sum(dim=1)
        denominator = torch.sqrt(
            pred_centered.square().sum(dim=1)
            * true_centered.square().sum(dim=1)
        ).clamp_min(self.eps)
        correlation = numerator / denominator
        downside_penalty = F.relu(-correlation).square().mean()
        rank_loss, rank_components = self.ranknet.forward_with_components(
            y_pred, y_true
        )
        self.last_components = {
            **self.ranknet.last_components,
            'stability_loss_raw': float(downside_penalty.detach()),
        }
        components = {
            **rank_components,
            'stability': downside_penalty,
        }
        return sum(components.values()), components

    def forward(self, y_pred, y_true):
        total_loss, _ = self.forward_with_components(y_pred, y_true)
        return total_loss


class HeadFocusedRankingLoss(nn.Module):
    """Top-K导向损失，并可惩罚分数对波动率指数的抄近路行为。"""

    def __init__(self, cfg, eps=1e-8):
        super().__init__()
        self.k = int(cfg.get('head_k', 5))
        self.head_weight = float(cfg.get('head_loss_weight', 1.0))
        self.global_weight = float(cfg.get('global_rank_weight', 0.2))
        self.stability_weight = float(cfg.get('stability_weight', 0.05))
        self.shortcut_weight = float(
            cfg.get('shortcut_correlation_weight', 0.0)
        )
        self.eps = eps

    @staticmethod
    def _mean_masked(loss, mask):
        counts = mask.sum(dim=(1, 2)).clamp_min(1)
        return ((loss * mask).sum(dim=(1, 2)) / counts).mean()

    def forward_with_components(self, y_pred, y_true, shortcut_index=None):
        num_items = y_true.size(1)
        k = min(self.k, num_items)
        threshold = torch.topk(y_true, k, dim=1).values[:, -1:]
        is_head = y_true >= threshold

        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)
        signed_loss = F.softplus(-pred_diff * torch.sign(true_diff))
        different = true_diff != 0

        # 有方向的“真实Top-K -> 非Top-K”比较；边界并列不会被任意拆开。
        head_vs_rest = (
            is_head.unsqueeze(2)
            & ~is_head.unsqueeze(1)
            & different
        )
        head_loss = self._mean_masked(signed_loss, head_vs_rest)

        # 小权重保留全截面相对监督，防止模型只拟合少数极端样本。
        global_loss = self._mean_masked(signed_loss, different)

        pred_centered = y_pred - y_pred.mean(dim=1, keepdim=True)
        true_centered = y_true.float() - y_true.float().mean(dim=1, keepdim=True)
        denominator = torch.sqrt(
            pred_centered.square().sum(dim=1)
            * true_centered.square().sum(dim=1)
        ).clamp_min(self.eps)
        correlation = (pred_centered * true_centered).sum(dim=1) / denominator
        stability_loss = F.relu(-correlation).square().mean()

        shortcut_correlation = y_pred.new_zeros(y_pred.size(0))
        if shortcut_index is not None:
            shortcut_index = shortcut_index.to(
                device=y_pred.device, dtype=y_pred.dtype
            )
            if shortcut_index.dim() == 1:
                shortcut_index = shortcut_index.unsqueeze(0)
            if shortcut_index.shape != y_pred.shape:
                raise ValueError(
                    'shortcut_index must have the same shape as y_pred'
                )
            shortcut_centered = shortcut_index - shortcut_index.mean(
                dim=1, keepdim=True
            )
            shortcut_denominator = torch.sqrt(
                pred_centered.square().sum(dim=1)
                * shortcut_centered.square().sum(dim=1)
            )
            valid_shortcut = shortcut_denominator > self.eps
            shortcut_correlation = torch.where(
                valid_shortcut,
                (pred_centered * shortcut_centered).sum(dim=1)
                / shortcut_denominator.clamp_min(self.eps),
                torch.zeros_like(shortcut_denominator),
            )
        shortcut_loss = shortcut_correlation.abs().mean()

        components = {
            'head': self.head_weight * head_loss,
            'global': self.global_weight * global_loss,
            'stability': self.stability_weight * stability_loss,
            'shortcut': self.shortcut_weight * shortcut_loss,
        }
        self.last_components = {
            'head_loss_raw': float(head_loss.detach()),
            'global_rank_loss_raw': float(global_loss.detach()),
            'stability_loss_raw': float(stability_loss.detach()),
            'shortcut_abs_correlation_raw': float(shortcut_loss.detach()),
            'shortcut_signed_correlation_raw': float(
                shortcut_correlation.mean().detach()
            ),
        }
        return sum(components.values()), components

    def forward(self, y_pred, y_true):
        total_loss, _ = self.forward_with_components(y_pred, y_true)
        return total_loss


def build_volatility_shortcut_index(
    sequences, feature_names, volatility_features, eps=1e-8
):
    """Create the equal-weight daily volatility index from the last step."""

    missing = [name for name in volatility_features if name not in feature_names]
    if missing:
        raise ValueError(
            f'shortcut volatility features missing from model inputs: {missing}'
        )
    positions = [feature_names.index(name) for name in volatility_features]
    values = sequences[:, -1, positions]
    centered = values - values.mean(dim=0, keepdim=True)
    scales = torch.sqrt(centered.square().mean(dim=0, keepdim=True))
    standardized = torch.where(
        scales > eps,
        centered / scales.clamp_min(eps),
        torch.zeros_like(centered),
    )
    return standardized.mean(dim=1).detach()


def apply_stability_weight_override(cfg, value):
    """Apply the CLI stability-weight override with shared validation."""

    if value is None:
        return cfg
    if value < 0:
        raise ValueError('stability-weight cannot be negative')
    cfg['stability_weight'] = float(value)
    return cfg


def build_ranking_criterion(cfg):
    loss_type = cfg.get('ranking_loss_type', 'weighted_ranknet')
    if loss_type == 'head_focused':
        return HeadFocusedRankingLoss(cfg)
    if loss_type == 'weighted_ranknet_stability':
        return RankNetStabilityLoss(cfg)
    if loss_type == 'weighted_ranknet':
        return WeightedRankingLoss(
            k=5,
            temperature=cfg.get('listwise_temperature', 1.0),
            target_type=cfg.get(
                'listwise_target_type', 'normalized_rank_softmax'
            ),
            weight_factor=cfg['top5_weight'],
            pairwise_weight=cfg['pairwise_weight'],
            base_weight=cfg.get('base_weight', 1.0),
        )
    raise ValueError(f"不支持的排序损失类型: {loss_type}")


def calculate_ranking_metrics(y_pred, y_true, masks, k=5):
    """计算Top-K组合收益、超额收益和头部排序质量。"""
    batch_size = y_pred.size(0)
    
    # Metrics accumulators
    pred_return_sum_list = []
    max_return_sum_list = []
    random_return_sum_list = []
    ratio_pred_list = []
    ratio_random_list = []
    final_score_list = []
    topk_returns = {size: [] for size in (5, 10, 20)}
    topk_excess_returns = {size: [] for size in (5, 10, 20)}
    ndcg5_list = []
    top5_percentile_list = []
    
    for i in range(batch_size):
        mask = masks[i]
        valid_indices = mask.nonzero().squeeze()
        
        if valid_indices.numel() < k:
            continue
            
        valid_pred = y_pred[i][valid_indices]
        valid_true = y_true[i][valid_indices] # This is the 5-day return
        # 1. Predicted Top 5
        _, pred_indices = torch.topk(valid_pred, k)
        pred_top_returns = valid_true[pred_indices]
        pred_return_sum = pred_top_returns.sum().item()
        
        # 2. True Top 5 (Theoretical Max)
        _, true_indices = torch.topk(valid_true, k)
        true_top_returns = valid_true[true_indices]
        max_return_sum = true_top_returns.sum().item()
        
        # 3. Random 5 (Expected Value)
        # Expected sum = 5 * mean(all valid returns)
        random_return_sum = k * valid_true.mean().item()
        
        # 计算每个样本的比例与稳定化 final_score
        ratio_pred = pred_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        ratio_random = random_return_sum / (max_return_sum + 1e-12) if abs(max_return_sum) > 1e-9 else 0.0
        denominator = max_return_sum - random_return_sum
        final_score = (pred_return_sum - random_return_sum) / (denominator + 1e-12) if abs(denominator) > 1e-6 else 0.0
        
        pred_return_sum_list.append(pred_return_sum)
        max_return_sum_list.append(max_return_sum)
        random_return_sum_list.append(random_return_sum)
        ratio_pred_list.append(ratio_pred)
        ratio_random_list.append(ratio_random)
        final_score_list.append(final_score)

        for size in topk_returns:
            actual_k = min(size, valid_true.numel())
            selected = torch.topk(valid_pred, actual_k).indices
            portfolio_return = valid_true[selected].mean().item()
            universe_return = valid_true.mean().item()
            topk_returns[size].append(portfolio_return)
            topk_excess_returns[size].append(
                portfolio_return - universe_return
            )

        # 连续百分位相关的NDCG@5：避免负收益作为gain导致不可解释的比值。
        actual_k = min(5, valid_true.numel())
        order = torch.argsort(valid_true)
        percentiles = torch.empty_like(valid_true, dtype=torch.float32)
        percentiles[order] = torch.linspace(
            0.0, 1.0, valid_true.numel(), device=valid_true.device
        )
        pred_top = torch.topk(valid_pred, actual_k).indices
        ideal_top = torch.topk(percentiles, actual_k).indices
        discounts = 1.0 / torch.log2(
            torch.arange(
                2, actual_k + 2, device=valid_true.device, dtype=torch.float32
            )
        )
        dcg = (percentiles[pred_top] * discounts).sum()
        idcg = (percentiles[ideal_top] * discounts).sum().clamp_min(1e-12)
        ndcg5_list.append(float((dcg / idcg).item()))
        top5_percentile_list.append(float(percentiles[pred_top].mean().item()))
        
    metrics = {
        'pred_return_sum': np.mean(pred_return_sum_list) if pred_return_sum_list else 0.0,
        'max_return_sum': np.mean(max_return_sum_list) if max_return_sum_list else 0.0,
        'random_return_sum': np.mean(random_return_sum_list) if random_return_sum_list else 0.0,
    }
    
    # 比值用逐样本均值，降低极端日影响
    metrics['ratio_pred'] = np.mean(ratio_pred_list) if ratio_pred_list else 0.0
    metrics['ratio_random'] = np.mean(ratio_random_list) if ratio_random_list else 0.0
    metrics['final_score'] = np.mean(final_score_list) if final_score_list else 0.0
    for size in topk_returns:
        metrics[f'top{size}_return'] = (
            np.mean(topk_returns[size]) if topk_returns[size] else 0.0
        )
        metrics[f'top{size}_excess_return'] = (
            np.mean(topk_excess_returns[size])
            if topk_excess_returns[size] else 0.0
        )
    metrics['ndcg_at_5'] = np.mean(ndcg5_list) if ndcg5_list else 0.0
    metrics['top5_true_percentile'] = (
        np.mean(top5_percentile_list) if top5_percentile_list else 0.0
    )
    return metrics

class RankingDataset(torch.utils.data.Dataset):
    """排序数据集类"""
    def __init__(self, sequences, targets, relevance_scores, stock_indices):
        self.sequences = sequences
        self.targets = targets
        self.relevance_scores = relevance_scores
        self.stock_indices = stock_indices
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'sequences': torch.FloatTensor(self.sequences[idx]),  # [num_stocks, seq_len, features]
            'targets': torch.FloatTensor(self.targets[idx]),      # [num_stocks] 真实涨跌幅
            'relevance': torch.FloatTensor(self.relevance_scores[idx]),  # 平均秩可含0.5
            'stock_indices': torch.LongTensor(self.stock_indices[idx])  # [num_stocks] 股票索引
        }

def collate_fn(batch):
    """自定义collate函数处理变长序列"""
    sequences = [item['sequences'] for item in batch]
    targets = [item['targets'] for item in batch]
    relevance = [item['relevance'] for item in batch]
    stock_indices = [item['stock_indices'] for item in batch]
    
    # 找到最大股票数量
    max_stocks = max(seq.size(0) for seq in sequences)
    
    # Padding到相同长度
    padded_sequences = []
    padded_targets = []
    padded_relevance = []
    padded_stock_indices = []
    masks = []
    
    for seq, tgt, rel, stock_idx in zip(sequences, targets, relevance, stock_indices):
        num_stocks = seq.size(0)
        seq_len = seq.size(1)
        feature_dim = seq.size(2)
        
        # 创建padding
        if num_stocks < max_stocks:
            pad_size = max_stocks - num_stocks
            seq_pad = torch.zeros(pad_size, seq_len, feature_dim)
            tgt_pad = torch.zeros(pad_size)
            rel_pad = torch.zeros(pad_size, dtype=torch.float32)
            stock_pad = torch.zeros(pad_size, dtype=torch.long)
            
            seq = torch.cat([seq, seq_pad], dim=0)
            tgt = torch.cat([tgt, tgt_pad], dim=0)
            rel = torch.cat([rel, rel_pad], dim=0)
            stock_idx = torch.cat([stock_idx, stock_pad], dim=0)
        
        # 创建mask标记有效位置
        mask = torch.ones(max_stocks)
        mask[num_stocks:] = 0
        
        padded_sequences.append(seq)
        padded_targets.append(tgt)
        padded_relevance.append(rel)
        padded_stock_indices.append(stock_idx)
        masks.append(mask)
    
    return {
        'sequences': torch.stack(padded_sequences),      # [batch, max_stocks, seq_len, features]
        'targets': torch.stack(padded_targets),          # [batch, max_stocks]
        'relevance': torch.stack(padded_relevance),      # [batch, max_stocks]
        'stock_indices': torch.stack(padded_stock_indices),  # [batch, max_stocks]
        'masks': torch.stack(masks)                      # [batch, max_stocks]
    }

# 排序训练函数
def train_ranking_model(
    model, dataloader, criterion, optimizer, device, epoch, writer,
    auxiliary_criterion=None, auxiliary_weight=0.0,
):
    model.train()
    total_loss = 0
    total_metrics = {}
    local_step = 0
    
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch+1}"):
        sequences = batch['sequences'].to(device)    # [batch, max_stocks, seq_len, features]
        targets = batch['targets'].to(device)        # [batch, max_stocks] 真实涨跌幅
        relevance = batch['relevance'].to(device)    # [batch, max_stocks] 预处理的相关性得分
        masks = batch['masks'].to(device)            # [batch, max_stocks] 有效位置mask
        
        optimizer.zero_grad()
        
        # 模型预测
        stock_padding_mask = ~masks.bool()
        outputs = model(
            sequences, stock_padding_mask=stock_padding_mask
        )  # [batch, max_stocks] 预测分数
        
        # 应用mask，只考虑有效股票
        masked_outputs = outputs * masks + (1 - masks) * (-1e9)  # 无效位置设为很小的值
        masked_targets = targets * masks
        masked_relevance = relevance.float() * masks  # 使用预处理好的相关性得分
        
        # 计算损失（只对有效股票计算）
        batch_loss = None
        component_rows = []
        batch_size = sequences.size(0)
        
        for i in range(batch_size):
            mask = masks[i]
            valid_indices = mask.nonzero().squeeze()
            
            if valid_indices.numel() == 0:
                continue
                
            if valid_indices.dim() == 0:
                valid_indices = valid_indices.unsqueeze(0)
            
            # 获取有效股票的预测值和预处理好的相关性得分
            valid_pred = masked_outputs[i][valid_indices]
            valid_relevance = masked_relevance[i][valid_indices]
            
            if len(valid_pred) > 1:
                # 排序损失
                if hasattr(criterion, 'forward_with_components'):
                    if isinstance(criterion, HeadFocusedRankingLoss):
                        shortcut_index = None
                        if criterion.shortcut_weight > 0:
                            shortcut_index = build_volatility_shortcut_index(
                                sequences[i][valid_indices],
                                config['feature_names'],
                                config['shortcut_volatility_features'],
                            )
                        ranking_loss, _ = criterion.forward_with_components(
                            valid_pred.unsqueeze(0),
                            valid_relevance.unsqueeze(0),
                            shortcut_index=(
                                shortcut_index.unsqueeze(0)
                                if shortcut_index is not None else None
                            ),
                        )
                    else:
                        ranking_loss, _ = criterion.forward_with_components(
                            valid_pred.unsqueeze(0),
                            valid_relevance.unsqueeze(0),
                        )
                else:
                    ranking_loss = criterion(
                        valid_pred.unsqueeze(0),
                        valid_relevance.unsqueeze(0),
                    )
                if hasattr(criterion, 'last_components'):
                    component_rows.append(criterion.last_components.copy())

                # 论文辅助回归损失；权重可设为0做消融，但组件必须保留。
                if auxiliary_criterion is not None and auxiliary_weight > 0:
                    valid_true = masked_targets[i][valid_indices]
                    aux_loss = auxiliary_criterion(
                        valid_pred.unsqueeze(0), valid_true.unsqueeze(0)
                    )
                    ranking_loss = ranking_loss + auxiliary_weight * aux_loss
                
                batch_loss = batch_loss + ranking_loss if isinstance(batch_loss, torch.Tensor) else ranking_loss
        
        if batch_loss is not None:
            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            grad_stats = clip_gradients(
                model.parameters(), config.get('max_grad_norm')
            )
            if writer:
                step = epoch * len(dataloader) + local_step
                writer.add_scalar(
                    'train/grad_norm_before_clip',
                    grad_stats['grad_norm_before_clip'],
                    global_step=step,
                )
                writer.add_scalar(
                    'train/gradient_clipped',
                    float(grad_stats['gradient_clipped']),
                    global_step=step,
                )
            optimizer.step()
            
            total_loss += batch_loss.item()
            
            # 计算评估指标
            with torch.no_grad():
                metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
                gate_values = getattr(model, 'last_gate_values', None)
                if gate_values is not None:
                    metrics['soft_gate_mean'] = float(gate_values.mean().cpu())
                if component_rows:
                    for name in component_rows[0]:
                        metrics[name] = float(np.mean(
                            [row[name] for row in component_rows if name in row]
                        ))
                metrics['grad_norm_before_clip'] = grad_stats[
                    'grad_norm_before_clip'
                ]
                metrics['gradient_clip_rate'] = float(
                    grad_stats['gradient_clipped']
                )
                metrics['nonfinite_gradient_count'] = 0.0
                for k, v in metrics.items():
                    if k not in total_metrics:
                        total_metrics[k] = 0
                    total_metrics[k] += v
            
            local_step += 1
            if writer:
                writer.add_scalar('train/loss', batch_loss.item(), global_step=epoch*len(dataloader)+local_step)
                for k, v in metrics.items():
                    writer.add_scalar(f'train/{k}', v, global_step=epoch*len(dataloader)+local_step)
    
    # 计算平均指标
    if local_step > 0:
        for k in total_metrics:
            total_metrics[k] /= local_step
    
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0, total_metrics

def evaluate_ranking_model(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    total_loss = 0
    total_metrics = {}
    num_batches = 0
    daily_ics = []  # RankIC仅作为Top-5主目标的辅助泛化诊断
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating Epoch {epoch+1}"):
            sequences = batch['sequences'].to(device)
            targets = batch['targets'].to(device)
            masks = batch['masks'].to(device)
            
            # 模型预测
            stock_padding_mask = ~masks.bool()
            outputs = model(
                sequences, stock_padding_mask=stock_padding_mask
            )
            
            # 应用mask
            masked_outputs = outputs * masks + (1 - masks) * (-1e9)
            masked_targets = targets * masks
            
            # 计算损失
            batch_loss = None
            batch_size = sequences.size(0)
            
            for i in range(batch_size):
                mask = masks[i]
                valid_indices = mask.nonzero().squeeze()
                
                if valid_indices.numel() == 0:
                    continue
                    
                if valid_indices.dim() == 0:
                    valid_indices = valid_indices.unsqueeze(0)
                
                valid_pred = masked_outputs[i][valid_indices]
                valid_true = masked_targets[i][valid_indices]
                if len(valid_pred) > 1:
                    relevance_scores = torch.as_tensor(
                        returns_to_relevance(valid_true.cpu().numpy()),
                        device=device,
                    )

                    loss = criterion(valid_pred.unsqueeze(0), relevance_scores.unsqueeze(0))
                    batch_loss = batch_loss + loss if batch_loss is not None else loss

                    # 逐日 Spearman RankIC：仅用有效股票的原始预测分数与真实收益
                    if valid_pred.numel() >= 10:
                        ic, _ = spearmanr(valid_pred.detach().cpu().numpy(), valid_true.detach().cpu().numpy())
                        if not np.isnan(ic):
                            daily_ics.append(float(ic))
            
            if batch_loss is not None:
                batch_loss = batch_loss / batch_size
                total_loss += batch_loss.item()
            
            # 计算评估指标
            metrics = calculate_ranking_metrics(masked_outputs, masked_targets, masks, k=5)
            gate_values = getattr(model, 'last_gate_values', None)
            if gate_values is not None:
                metrics['soft_gate_mean'] = float(gate_values.mean().cpu())
            for k, v in metrics.items():
                if k not in total_metrics:
                    total_metrics[k] = 0
                total_metrics[k] += v

            num_batches += 1

    # 计算平均指标
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    for k in total_metrics:
        total_metrics[k] /= num_batches

    # 逐日 Spearman RankIC 汇总（辅助诊断，不参与主选模）
    total_metrics.update(summarize_daily_ics(daily_ics))
    if writer:
        writer.add_scalar('eval/loss', avg_loss, global_step=epoch)
        for k, v in total_metrics.items():
            writer.add_scalar(f'eval/{k}', v, global_step=epoch)
    
    return avg_loss, total_metrics


def predict_top_stocks(model, data, features, sequence_length, scaler, stockid2idx, device, top_k=5):
    """
    预测某一天涨幅前top_k的股票
    """
    model.eval()
    
    # 获取最后一天的数据作为预测基础
    latest_date = data['日期'].max()
    
    # 准备预测数据
    day_sequences = []
    day_stock_codes = []
    day_stock_indices = []
    
    for stock_code in data['股票代码'].unique():
        # 获取该股票历史sequence_length天的数据
        stock_history = data[
            (data['股票代码'] == stock_code) & 
            (data['日期'] <= latest_date)
        ].sort_values('日期').tail(sequence_length)
        
        if len(stock_history) == sequence_length:
            seq = stock_history[features].values
            day_sequences.append(seq)
            day_stock_codes.append(stock_code)
            day_stock_indices.append(stockid2idx[stock_code])
    
    if len(day_sequences) == 0:
        return []
    
    # 转换为tensor
    sequences = torch.FloatTensor(np.array(day_sequences)).unsqueeze(0).to(device)  # [1, num_stocks, seq_len, features]
    
    with torch.no_grad():
        # 模型预测
        outputs = model(sequences)  # [1, num_stocks]
        scores = outputs.squeeze().cpu().numpy()  # [num_stocks]
        
        # 获取排名前top_k的股票
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        top_stocks = []
        for idx in top_indices:
            top_stocks.append({
                'stock_code': day_stock_codes[idx],
                'predicted_score': scores[idx],
                'rank': len(top_stocks) + 1
            })
    
    return top_stocks

def save_predictions(top_stocks, output_path):
    """保存预测结果"""
    results = []
    for stock in top_stocks:
        results.append({
            '排名': stock['rank'],
            '股票代码': stock['stock_code'],
            '预测分数': stock['predicted_score']
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"预测结果已保存到: {output_path}")


def split_train_val_by_last_month(df, sequence_length):
    """按最后一个月做验证集划分，并为验证集补充序列上下文。"""
    df = df.copy()
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['日期', '股票代码']).reset_index(drop=True)

    last_date = df['日期'].max()
    val_start = (last_date - pd.DateOffset(months=2)).normalize()

    # 验证集需要保留前 sequence_length-1 个交易日作为序列上下文，
    # 这样第一个验证样本的窗口结束日就可以落在 val_start。
    val_context_start = val_start - pd.tseries.offsets.BDay(sequence_length - 1)

    train_df = df[df['日期'] < val_start].copy()
    val_df = df[df['日期'] >= val_context_start].copy()

    print(f"全量数据范围: {df['日期'].min().date()} 到 {last_date.date()}")
    print(f"训练集范围: {train_df['日期'].min().date()} 到 {train_df['日期'].max().date()}")
    print(f"验证集目标范围(最后一个月): {val_start.date()} 到 {last_date.date()}")
    print(f"验证集实际取数范围(含序列上下文): {val_df['日期'].min().date()} 到 {val_df['日期'].max().date()}")

    # 恢复为字符串，保持与原流程一致
    train_df['日期'] = train_df['日期'].dt.strftime('%Y-%m-%d')
    val_df['日期'] = val_df['日期'].dt.strftime('%Y-%m-%d')

    return train_df, val_df, val_start


def split_dates_train_val_test(
    df, train_days=404, gap_days=5, val_days=50, test_days=60,
    gap2_days=None,
):
    """按交易日序号将数据划分为 train/gap/val/gap/test 五段。

    gap1使用gap_days；gap2可独立设置，默认与gap1相同。
    两处隔离均防止未来5日标签跨段泄漏。
    返回各段起止日期与交易日数，供训练与评测复用（保存到 data_split.json）。
    """
    dates = sorted(pd.to_datetime(df['日期']).dt.normalize().unique())
    gap2_days = gap_days if gap2_days is None else int(gap2_days)
    if gap2_days < 0:
        raise ValueError("gap2_days不能为负数")
    n = len(dates)
    if test_days < 0:
        raise ValueError("test_days不能为负数")
    need = train_days + gap_days + val_days
    if test_days > 0:
        need += gap2_days + test_days
    if n < need:
        raise ValueError(f"交易日不足：需要 {need} 天，实际 {n} 天")

    i_tr_end = train_days - 1
    i_val_start = train_days + gap_days
    i_val_end = i_val_start + val_days - 1

    def d(i):
        return pd.Timestamp(dates[i]).strftime('%Y-%m-%d')

    split = {
        'split_method': 'trading_day_index',
        'total_trading_days': int(n),
        'label_horizon': 5,
        'train': {'start': d(0), 'end': d(i_tr_end), 'n_days': train_days},
        'gap1': {'start': d(i_tr_end + 1), 'end': d(i_val_start - 1), 'n_days': gap_days},
        'validation': {'start': d(i_val_start), 'end': d(i_val_end), 'n_days': val_days},
    }
    if test_days > 0:
        i_te_start = i_val_end + 1 + gap2_days
        i_te_end = i_te_start + test_days - 1
        split['gap2'] = {
            'start': d(i_val_end + 1),
            'end': d(i_te_start - 1),
            'n_days': gap2_days,
        }
        split['test'] = {
            'start': d(i_te_start), 'end': d(i_te_end), 'n_days': test_days
        }
    else:
        split['gap2'] = {'start': None, 'end': None, 'n_days': 0}
        split['test'] = {'start': None, 'end': None, 'n_days': 0}
    return split


def restrict_split_source_to_recent_years(df, years):
    """将目标日期限制为最新交易日前推若干自然年（两端均包含）。"""
    if years is None:
        return df.copy(), None
    if years <= 0:
        raise ValueError("data_lookback_years必须为正数")
    normalized_dates = pd.to_datetime(df['日期']).dt.normalize()
    latest_date = normalized_dates.max()
    cutoff_date = latest_date - pd.DateOffset(years=int(years))
    selected = df.loc[normalized_dates >= cutoff_date].copy()
    if selected.empty:
        raise ValueError("最近年份窗口内没有可用于切分的交易日")
    return selected, {
        'lookback_years': int(years),
        'cutoff_date': cutoff_date.strftime('%Y-%m-%d'),
        'first_trading_date': pd.to_datetime(selected['日期']).min().strftime('%Y-%m-%d'),
        'latest_trading_date': latest_date.strftime('%Y-%m-%d'),
        'n_trading_days': int(pd.to_datetime(selected['日期']).dt.normalize().nunique()),
    }


def ratio_split_day_counts(df, ratios=(0.70, 0.15, 0.15), gap_days=5):
    """扣除两处gap后，按比例计算train/validation/test交易日数。"""
    if len(ratios) != 3 or any(float(ratio) <= 0 for ratio in ratios):
        raise ValueError("split_ratios必须包含三个正数")
    ratio_sum = float(sum(ratios))
    normalized_ratios = [float(ratio) / ratio_sum for ratio in ratios]
    n_days = int(pd.to_datetime(df['日期']).dt.normalize().nunique())
    usable_days = n_days - 2 * int(gap_days)
    if usable_days < 3:
        raise ValueError("扣除两处gap后交易日不足，无法按三段切分")
    train_days = round(usable_days * normalized_ratios[0])
    val_days = round(usable_days * normalized_ratios[1])
    test_days = usable_days - train_days - val_days
    if min(train_days, val_days, test_days) <= 0:
        raise ValueError("比例切分后存在空数据段")
    return train_days, val_days, test_days


def split_dates_competition_5day(df, train_days=None, train_end_date=None):
    """单次比赛口径：一个信号日，随后五个评分交易日，无验证集。

    训练标签截止日距离信号日五个交易日，使最后一个训练标签的退出价
    最晚落在信号日；评分窗口信息不会进入训练。
    """
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(df['日期']).dt.normalize().unique()))
    if train_days is not None and train_end_date is not None:
        raise ValueError('competition_train_days与competition_train_end_date不能同时设置')
    if train_days is not None:
        train_days = int(train_days)
        if train_days <= 0:
            raise ValueError('competition_train_days必须为正数')
        required_days = train_days + 10
        if len(dates) < required_days:
            raise ValueError(
                f'比赛滚动窗口需要{required_days}个交易日，实际仅有{len(dates)}个'
            )
        dates = dates[-required_days:]
    if len(dates) < 12:
        raise ValueError("交易日不足，无法构造训练集和单次5日比赛窗口")
    score_dates = dates[-5:]
    signal_date = dates[-6]
    if train_end_date is None:
        train_dates = dates[:-10]
    else:
        train_end_date = pd.Timestamp(train_end_date).normalize()
        train_dates = dates[dates <= train_end_date]
        if len(train_dates) == 0:
            raise ValueError('competition_train_end_date之前没有训练交易日')
        train_end_position = int(dates.get_loc(train_dates[-1]))
        signal_position = int(dates.get_loc(signal_date))
        if signal_position - train_end_position < 5:
            raise ValueError(
                '训练截止日与信号日之间不足5个交易日，最后训练标签可能跨入评分阶段'
            )
    # 保持既有competition_5day审计口径：隔离上下文包含信号日；信号日
    # 同时单列，表示该日特征可用于生成排名，但不能作为训练目标。
    embargo_dates = dates[(dates > train_dates[-1]) & (dates <= signal_date)]
    if len(embargo_dates) < 5:
        raise ValueError('训练截止日与信号日之间的隔离交易日不足5日')
    return {
        'split_method': 'competition_single_signal_5day_score',
        'window_trading_days': int(len(dates)),
        'total_trading_days': int(len(dates)),
        'label_horizon': 5,
        'validation': {'start': None, 'end': None, 'n_days': 0},
        'train': {
            'start': train_dates[0].strftime('%Y-%m-%d'),
            'end': train_dates[-1].strftime('%Y-%m-%d'),
            'n_days': int(len(train_dates)),
        },
        'embargo_context': {
            'start': embargo_dates[0].strftime('%Y-%m-%d'),
            'end': embargo_dates[-1].strftime('%Y-%m-%d'),
            'n_days': int(len(embargo_dates)),
            'note': 'not training targets; retained as known sequence context',
        },
        'signal': {
            'date': signal_date.strftime('%Y-%m-%d'),
            'n_days': 1,
        },
        'test': {
            'start': score_dates[0].strftime('%Y-%m-%d'),
            'end': score_dates[-1].strftime('%Y-%m-%d'),
            'n_days': 5,
            'role': 'score_self.py weighted return window',
        },
    }


# 主程序
def main():
    set_seed(config.get('seed', 42))
    output_dir = config['output_dir']
    os.makedirs(output_dir,exist_ok=True)
    # 保存在output_dir中保存当前的配置文件，以便复现
    data_path = config['data_path']
    # config.json 会在 INFO 搜索完成后再次保存，包含最终参数
    os.makedirs(output_dir, exist_ok=True)
    is_train = True
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'log')) if is_train else None
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # 1. 数据加载
    data_file = config.get('data_file') or os.path.join(data_path, 'train.csv')
    if not os.path.isabs(data_file):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_file = os.path.join(project_root, data_file)
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"未找到训练数据: {data_file}。PIT实验必须先运行 data/build_hs300_pit.py。"
        )
    full_df = pd.read_csv(data_file, dtype={'股票代码': str})
    full_df['股票代码'] = full_df['股票代码'].astype(str).str.zfill(6)
    for boolean_column in ['is_member', 'is_suspended', 'is_tradable', 'in_signal_range']:
        if boolean_column in full_df.columns:
            full_df[boolean_column] = (
                full_df[boolean_column].astype(str).str.lower().isin({'true', '1'})
            )
    dataset_manifest = os.path.join(os.path.dirname(data_file), 'manifest.json')
    if 'is_member' in full_df.columns:
        if not os.path.exists(dataset_manifest):
            raise FileNotFoundError("PIT模型数据缺少同目录 manifest.json，拒绝训练")
        with open(dataset_manifest, 'r', encoding='utf-8') as handle:
            manifest_payload = json.load(handle)
        if manifest_payload.get('status') != 'validated':
            raise ValueError("PIT数据 manifest 状态不是 validated，拒绝训练")
        config['dataset_file'] = data_file
        config['dataset_sha256'] = sha256_file(data_file)
        config['dataset_manifest'] = dataset_manifest
        config['dataset_manifest_sha256'] = sha256_file(dataset_manifest)
        config['dataset_kind'] = 'csi300_point_in_time'
        if 'in_signal_range' in full_df.columns:
            split_source = full_df[full_df['in_signal_range']].copy()
        else:
            split_source = full_df[full_df['is_member'].astype(bool)].copy()
    else:
        config['dataset_kind'] = 'legacy_fixed_universe'
        split_source = full_df

    split_source, recent_window = restrict_split_source_to_recent_years(
        split_source, config.get('data_lookback_years')
    )
    if recent_window is not None:
        config['data_window'] = recent_window
    split_start_date = config.get('split_start_date')
    if split_start_date:
        split_start_date = pd.Timestamp(split_start_date).normalize()
        split_source = split_source[
            pd.to_datetime(split_source['日期']).dt.normalize() >= split_start_date
        ].copy()
        if split_source.empty:
            raise ValueError(
                f"split_start_date={split_start_date.date()}之后没有可切分数据"
            )
        config['split_start_date'] = split_start_date.strftime('%Y-%m-%d')
    window_end_date = config.get('window_end_date')
    if window_end_date:
        window_end_date = pd.Timestamp(window_end_date).normalize()
        split_source = split_source[
            pd.to_datetime(split_source['日期']).dt.normalize() <= window_end_date
        ].copy()
        if split_source.empty:
            raise ValueError(
                f"window_end_date={window_end_date.date()}之前没有可切分数据"
            )
        config['window_end_date'] = window_end_date.strftime('%Y-%m-%d')

    # 获取所有股票ID，建立映射
    all_stock_ids = full_df['股票代码'].unique()
    stockid2idx = {sid: idx for idx, sid in enumerate(sorted(all_stock_ids))}
    num_stocks = len(stockid2idx)

    # 2. 比赛阶段只保留训练集和单次5日评分窗口；论文阶段仍可使用三段协议。
    competition_5day = config.get('training_protocol') == 'competition_5day'
    gap_days = config.get('gap_days', 5)
    if competition_5day:
        if gap_days != 5:
            raise ValueError('competition_5day协议固定使用5个交易日gap')
        split = split_dates_competition_5day(
            split_source,
            train_days=config.get('competition_train_days'),
            train_end_date=config.get('competition_train_end_date'),
        )
        config['train_days'] = split['train']['n_days']
        config['val_days'] = 0
        config['test_days'] = 5
    else:
        if config.get('split_ratios') and not config.get('_explicit_split_days', False):
            train_days, val_days, test_days = ratio_split_day_counts(
                split_source,
                ratios=config['split_ratios'],
                gap_days=gap_days,
            )
            config['train_days'] = train_days
            config['val_days'] = val_days
            config['test_days'] = test_days
        split = split_dates_train_val_test(
            split_source,
            train_days=config.get('train_days', 404),
            gap_days=gap_days,
            val_days=config.get('val_days', 50),
            test_days=config.get('test_days', 60),
            gap2_days=config.get('gap2_days'),
        )
    train_start = pd.to_datetime(split['train']['start'])
    train_end = pd.to_datetime(split['train']['end'])
    if competition_5day:
        val_start = pd.to_datetime(split['signal']['date'])
        val_end = val_start
    else:
        val_start = pd.to_datetime(split['validation']['start'])
        val_end = pd.to_datetime(split['validation']['end'])
    with open(os.path.join(output_dir, 'data_split.json'), 'w', encoding='utf-8') as f:
        json.dump(split, f, indent=2, ensure_ascii=False)
    print("数据切分（交易日）:")
    if competition_5day:
        print(f"  train      : {split['train']['start']} ~ {split['train']['end']}  ({split['train']['n_days']}天)")
        print(f"  signal     : {split['signal']['date']}  (1天)")
        print(f"  score_5day : {split['test']['start']} ~ {split['test']['end']}  (5天)")
    else:
        for seg in ['train', 'gap1', 'validation', 'gap2', 'test']:
            s = split[seg]
            print(f"  {seg:11s}: {s['start']} ~ {s['end']}  ({s['n_days']}天)")

    # 3. 全时间线一次性特征工程（所有特征均为滞后特征，先全局计算再按日期归属，符合P0-3）
    processed_all, features = preprocess_data(full_df, is_train=True, stockid2idx=stockid2idx)
    excluded = set(config.get('exclude_features', []))
    missing_exclusions = excluded.difference(features)
    if missing_exclusions:
        raise ValueError(f"待删除特征不存在: {sorted(missing_exclusions)}")
    features = [feature for feature in features if feature not in excluded]
    included = config.get('include_features')
    if included:
        missing_inclusions = set(included).difference(features)
        if missing_inclusions:
            raise ValueError(f"指定特征不存在或已被删除: {sorted(missing_inclusions)}")
        features = list(included)
    base_features = [feature for feature in features if feature in BASELINE24_FEATURES]
    market_features = [feature for feature in features if feature in MARKET4_FEATURES]
    config['base_feature_names'] = list(base_features)
    config['market_feature_names'] = list(market_features)
    processed_all['日期'] = pd.to_datetime(processed_all['日期'])

    # 横截面策略：每天只使用当天全部股票的当期特征做百分位排名。
    # 原特征已由截至当天的历史数据构造，因此这里不使用未来信息。
    if config.get('feature_transform') == 'cross_sectional_rank':
        if 'is_member' in processed_all.columns:
            processed_all = point_in_time_cross_sectional_rank(
                processed_all, base_features
            )
        else:
            processed_all[base_features] = (
                processed_all.groupby('日期', sort=False)[base_features]
                .rank(method='average', pct=True)
            )

    # 显式交叉项必须在个股特征完成横截面排名之后构造。
    # 市场压力的中心和尺度只拟合训练日，随配置快照供推理复用。
    if config['feature_num'] in {
        'baseline24_market4_cross4', 'baseline24_market4_component2'
    }:
        cross_fit_rows = (
            (processed_all['日期'] >= train_start)
            & (processed_all['日期'] <= train_end)
        )
        if 'is_member' in processed_all.columns:
            cross_fit_rows &= processed_all['is_member'].fillna(False).astype(bool)
        if 'is_tradable' in processed_all.columns:
            cross_fit_rows &= processed_all['is_tradable'].fillna(False).astype(bool)
        if 'in_signal_range' in processed_all.columns:
            cross_fit_rows &= processed_all['in_signal_range'].fillna(False).astype(bool)
        _, cross_center, cross_scale = fit_market_stress_statistics(
            processed_all, cross_fit_rows
        )
        config['market_cross_center_raw'] = cross_center.to_numpy().tolist()
        config['market_cross_scale_raw'] = cross_scale.to_numpy().tolist()
        if config['feature_num'] == 'baseline24_market4_cross4':
            cross_features = list(CROSS4_FEATURES)
            processed_all = add_market_cross_features(
                processed_all,
                config['market_cross_center_raw'],
                config['market_cross_scale_raw'],
            )
            cross_formula = 'combined_market_stress_x_four_stock_factors'
        else:
            cross_features = list(COMPONENT2_FEATURES)
            processed_all = add_market_component_cross_features(
                processed_all,
                config['market_cross_center_raw'],
                config['market_cross_scale_raw'],
            )
            cross_formula = (
                'z(market_volatility_20)*ROC20_rank; '
                '-z(market_drawdown_60)*STD20_rank'
            )
        config['market_cross_features'] = cross_features
        with open(
            os.path.join(output_dir, 'market_cross_features.json'),
            'w',
            encoding='utf-8',
        ) as handle:
            json.dump({
                'fit_period': [split['train']['start'], split['train']['end']],
                'market_features': list(MARKET4_FEATURES),
                'market_center_raw': cross_center.to_dict(),
                'market_scale_raw': cross_scale.to_dict(),
                'stress_directions': [-1.0, 1.0, -1.0, -1.0],
                'cross_features': cross_features,
                'cross_formula': cross_formula,
                'low_volatility_features': list(
                    config.get('soft_gate_low_volatility_features', [])
                ),
            }, handle, indent=2, ensure_ascii=False)

    config['feature_names'] = list(features)

    # 4. 标准化：Scaler 仅在 train 段拟合，再 transform 全部（防止val/test统计量泄露）
    processed_all[features] = processed_all[features].replace([np.inf, -np.inf], np.nan)
    processed_all = processed_all.dropna(subset=features)
    train_rows = (
        (processed_all['日期'] >= train_start)
        & (processed_all['日期'] <= train_end)
    )
    if 'is_member' in processed_all.columns:
        train_rows &= processed_all['is_member'].fillna(False).astype(bool)
    if 'is_tradable' in processed_all.columns:
        train_rows &= processed_all['is_tradable'].fillna(False).astype(bool)
    if 'in_signal_range' in processed_all.columns:
        train_rows &= processed_all['in_signal_range'].fillna(False).astype(bool)
    if not train_rows.any():
        raise ValueError("训练区间没有PIT可交易成分样本，拒绝拟合Scaler")
    reuse_scaler_file = config.get('reuse_scaler_file')
    if reuse_scaler_file:
        scaler = joblib.load(reuse_scaler_file)
        if getattr(scaler, 'n_features_in_', len(features)) != len(features):
            raise ValueError("复用Scaler的特征数与当前模型不一致")
        print(f"近期微调复用长期训练Scaler: {reuse_scaler_file}")
    else:
        scaler = StandardScaler()
        scaler.fit(processed_all.loc[train_rows, features])
    if (
        config.get('soft_gate_enabled', False)
        and config.get('soft_gate_rule_mode', False)
    ):
        if market_features != list(config.get('soft_gate_market_features', [])):
            raise ValueError('rule gate requires the configured four market features in order')
        date_columns = [
            column for column in processed_all.columns
            if pd.api.types.is_datetime64_any_dtype(processed_all[column])
        ]
        if len(date_columns) != 1:
            raise ValueError(f'rule gate expected one datetime column, got {date_columns}')
        date_column = date_columns[0]
        daily_market = (
            processed_all.loc[train_rows, [date_column] + market_features]
            .groupby(date_column, sort=True)[market_features]
            .mean()
        )
        reuse_gate_rule_file = config.get('reuse_soft_gate_rule_file')
        reused_gate_rule = None
        if reuse_gate_rule_file:
            with open(reuse_gate_rule_file, 'r', encoding='utf-8') as handle:
                reused_gate_rule = json.load(handle)
            if reused_gate_rule.get('market_features') != market_features:
                raise ValueError('复用门控规则的市场特征及顺序与当前配置不一致')
            gate_mean_raw = pd.Series(
                reused_gate_rule['market_mean_raw'], dtype=float
            ).reindex(market_features)
            gate_std_raw = pd.Series(
                reused_gate_rule['market_std_raw'], dtype=float
            ).reindex(market_features)
        else:
            gate_mean_raw = daily_market.mean(axis=0)
            gate_std_raw = daily_market.std(axis=0, ddof=0)
        if (gate_std_raw <= 0).any() or gate_std_raw.isna().any():
            raise ValueError('rule gate market feature has invalid training-time scale')
        feature_positions = [features.index(name) for name in market_features]
        scaler_mean = np.asarray(scaler.mean_)[feature_positions]
        scaler_scale = np.asarray(scaler.scale_)[feature_positions]
        config['soft_gate_market_center'] = (
            (gate_mean_raw.to_numpy() - scaler_mean) / scaler_scale
        ).tolist()
        config['soft_gate_market_scale'] = (
            gate_std_raw.to_numpy() / scaler_scale
        ).tolist()
        stress_directions = np.asarray([-1.0, 1.0, -1.0, -1.0])
        low_q = float(
            reused_gate_rule.get('pressure_low_quantile', 0.70)
            if reused_gate_rule else
            config.get('soft_gate_threshold_low_quantile', 0.70)
        )
        high_q = float(
            reused_gate_rule.get('pressure_high_quantile', 0.90)
            if reused_gate_rule else
            config.get('soft_gate_threshold_high_quantile', 0.90)
        )
        if not 0.0 < low_q < high_q < 1.0:
            raise ValueError('soft gate threshold quantiles must satisfy 0 < low < high < 1')
        daily_stress_components = (
            ((daily_market - gate_mean_raw) / gate_std_raw).to_numpy()
            * stress_directions
        )
        if reused_gate_rule:
            config['soft_gate_threshold_low'] = float(
                reused_gate_rule['pressure_threshold_low']
            )
            config['soft_gate_threshold_high'] = float(
                reused_gate_rule['pressure_threshold_high']
            )
            component_low = reused_gate_rule.get('component_threshold_low')
            component_high = reused_gate_rule.get('component_threshold_high')
            if component_low is not None and component_high is not None:
                config['soft_gate_component_threshold_low'] = [
                    float(component_low[name]) for name in market_features
                ]
                config['soft_gate_component_threshold_high'] = [
                    float(component_high[name]) for name in market_features
                ]
            elif config.get('soft_gate_trigger_mode') == 'two_of_four':
                raise ValueError(
                    'two_of_four gate cannot reuse a legacy rule without component thresholds'
                )
        else:
            daily_pressure = daily_stress_components.mean(axis=1)
            config['soft_gate_threshold_low'] = float(
                np.quantile(daily_pressure, low_q)
            )
            config['soft_gate_threshold_high'] = float(
                np.quantile(daily_pressure, high_q)
            )
            config['soft_gate_component_threshold_low'] = np.quantile(
                daily_stress_components, low_q, axis=0
            ).tolist()
            config['soft_gate_component_threshold_high'] = np.quantile(
                daily_stress_components, high_q, axis=0
            ).tolist()
        with open(os.path.join(output_dir, 'soft_gate_rule.json'), 'w', encoding='utf-8') as handle:
            json.dump({
                'fit_period': [split['train']['start'], split['train']['end']],
                'daily_observations': int(len(daily_market)),
                'market_features': market_features,
                'market_mean_raw': gate_mean_raw.to_dict(),
                'market_std_raw': gate_std_raw.to_dict(),
                'pressure_low_quantile': low_q,
                'pressure_high_quantile': high_q,
                'pressure_threshold_low': config['soft_gate_threshold_low'],
                'pressure_threshold_high': config['soft_gate_threshold_high'],
                'component_threshold_low': dict(zip(
                    market_features,
                    config.get('soft_gate_component_threshold_low', []),
                )),
                'component_threshold_high': dict(zip(
                    market_features,
                    config.get('soft_gate_component_threshold_high', []),
                )),
                'trigger_mode': config.get('soft_gate_trigger_mode', 'aggregate'),
                'minimum_abnormal_components': int(
                    config.get('soft_gate_min_abnormal_components', 2)
                ),
                'rule_min': float(config.get('soft_gate_rule_min', 0.3)),
                'rule_max': float(config.get('soft_gate_rule_max', 0.5)),
                'gate_cap': float(config.get('soft_gate_cap', 0.6)),
                'residual_fraction': float(config.get('soft_gate_residual_fraction', 0.2)),
                'reused_rule_source': reuse_gate_rule_file,
                'source_fit_period': (
                    reused_gate_rule.get('fit_period') if reused_gate_rule else None
                ),
            }, handle, indent=2, ensure_ascii=False)
    processed_unscaled = processed_all.copy()
    processed_all[features] = scaler.transform(processed_all[features])
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

    # 5. 构建 train / val 每日横截面样本（test 段训练阶段不使用，严格隔离）
    train_result = create_ranking_dataset_vectorized(
        processed_all,
        features,
        config['sequence_length'],
        min_window_end_date=train_start,
        max_window_end_date=train_end,
        return_dates=True,
    )
    train_sequences, train_targets, train_relevance, train_stock_indices, train_sample_dates = train_result
    val_sequences, val_targets, val_relevance, val_stock_indices = create_ranking_dataset_vectorized(
        processed_all,
        features,
        config['sequence_length'],
        min_window_end_date=val_start,
        max_window_end_date=val_end,
    )

    test_dataset = None
    if not competition_5day and split['test']['n_days'] > 0:
        test_sequences, test_targets, test_relevance, test_stock_indices = (
            create_ranking_dataset_vectorized(
                processed_all,
                features,
                config['sequence_length'],
                min_window_end_date=pd.to_datetime(split['test']['start']),
                max_window_end_date=pd.to_datetime(split['test']['end']),
            )
        )
        test_dataset = RankingDataset(
            test_sequences, test_targets, test_relevance, test_stock_indices
        )
        print(f"测试集样本数: {len(test_sequences)}")

    print(f"训练集样本数: {len(train_sequences)}")
    print(f"{'比赛信号日' if competition_5day else '验证集'}样本数: {len(val_sequences)}")
    
    # 5. 创建排序数据集和数据加载器
    train_dataset = RankingDataset(train_sequences, train_targets, train_relevance, train_stock_indices)
    val_dataset = RankingDataset(val_sequences, val_targets, val_relevance, val_stock_indices)
    
    train_sampler = None
    if config.get('sampling_strategy', 'natural') == 'stratified_regime':
        state_source = full_df
        if 'is_member' in state_source.columns:
            state_source = state_source[state_source['is_member'].fillna(False).astype(bool)]
        state_table = build_causal_market_state_table(state_source)
        thresholds = fit_market_state_thresholds(state_table, train_sample_dates)
        state_table = apply_market_state_labels(state_table, thresholds)
        state_lookup = state_table.set_index('日期')['market_state']
        train_dates_index = pd.DatetimeIndex(train_sample_dates).normalize()
        train_states = state_lookup.reindex(train_dates_index)
        if train_states.isna().any():
            missing_dates = train_dates_index[train_states.isna()]
            raise ValueError(f"市场状态表缺少训练日期: {list(missing_dates[:5])}")
        state_table['is_threshold_fit_train_date'] = state_table['日期'].isin(train_dates_index)
        state_table.to_csv(os.path.join(output_dir, 'market_state_audit.csv'), index=False)
        with open(os.path.join(output_dir, 'market_state_thresholds.json'), 'w', encoding='utf-8') as f:
            json.dump(thresholds, f, indent=2, ensure_ascii=False)
        sampling_audit_file = os.path.join(output_dir, 'sampling_audit.jsonl')
        if os.path.exists(sampling_audit_file):
            os.remove(sampling_audit_file)
        train_sampler = AuditedStratifiedDateSampler(
            sample_dates=train_sample_dates,
            states=train_states.tolist(),
            weak_target_share=config.get('weak_target_share', 0.30),
            seed=config.get('seed', 42),
            audit_file=sampling_audit_file,
        )
        config['market_state_thresholds'] = thresholds
        config['natural_train_state_counts'] = {
            str(k): int(v) for k, v in train_states.value_counts().to_dict().items()
        }

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0,  # 减少worker数量避免内存问题
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False,
        )
    
    # 6. 论文INFO外层超参优化。默认关闭，只有显式启用时运行。
    if config.get('use_info', False):
        def build_model(cfg):
            return INFO_TCN_iTransformer(
                input_dim=len(features), config=cfg, num_stocks=num_stocks
            )

        info_opt = INFOOptimizer(
            base_config=config,
            search_budget=config.get('info_search_budget', 10),
            num_trials=config.get('info_num_trials', 3),
            lambda_stability=config.get('info_stability_lambda', 0.5),
            max_epochs=config.get('info_max_epochs', 10),
            seed=config.get('seed', 42),
        )
        info_criterion = build_ranking_criterion(config)
        best_config = info_opt.search(
            train_fn=train_ranking_model,
            eval_fn=evaluate_ranking_model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_builder=build_model,
            device=device,
            criterion=info_criterion,
            verbose=True,
        )
        with open(
            os.path.join(output_dir, 'info_search_report.txt'),
            'w',
            encoding='utf-8',
        ) as f:
            f.write(info_opt.get_search_summary())
            f.write("\n\n搜索历史:\n")
            for i, (cfg, scores, iscore) in enumerate(info_opt.results):
                f.write(f"Config {i+1}: INFO={iscore:.4f}, scores={scores}\n")

        if (
            config.get('info_require_positive', False)
            and info_opt.best_info_score <= 0
        ):
            raise RuntimeError(
                "INFO搜索的全部候选INFO Score均<=0，按配置拒绝进入正式训练"
            )

        searchable_keys = [
            'learning_rate', 'tcn_num_layers', 'tcn_channels',
            'it_num_layers', 'dropout',
        ]
        for key in searchable_keys:
            if key in best_config:
                config[key] = best_config[key]

    # 7. 用冻结或INFO选定参数构建最终模型
    model = INFO_TCN_iTransformer(input_dim=len(features), config=config, num_stocks=num_stocks)
    init_checkpoint = config.get('init_checkpoint')
    if init_checkpoint:
        model.load_state_dict(torch.load(init_checkpoint, map_location=device))
        print(f"已加载初始化checkpoint: {init_checkpoint}")
    model.to(device)
    print(f"\n最终模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 8. 损失函数和优化器
    criterion = build_ranking_criterion(config)

    auxiliary_criterion = AuxiliaryRegressionLoss(alpha=0.5)
    auxiliary_weight = config.get('auxiliary_loss_weight', 0.0)
    
    if config.get('soft_gate_enabled', False):
        gate_parameters = list(model.market_gate.parameters())
        if not config.get('soft_gate_rule_mode', False):
            gate_parameters.append(model.soft_gate_bias)
        gate_parameter_ids = {id(parameter) for parameter in gate_parameters}
        base_parameters = [
            parameter for parameter in model.parameters()
            if id(parameter) not in gate_parameter_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {'params': base_parameters, 'lr': config['learning_rate']},
                {
                    'params': gate_parameters,
                    'lr': config['learning_rate']
                    * float(config.get('soft_gate_learning_rate_multiplier', 20.0)),
                },
            ],
            weight_decay=1e-5,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config['learning_rate'], weight_decay=1e-5
        )
    scheduler = build_linear_scheduler(optimizer, config)
    
    # 9. 保存最终配置快照（INFO 搜索后的参数）
    with open(os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump({k: v for k, v in config.items() if not k.startswith('_')},
                  f, indent=4, ensure_ascii=False, default=str)
    target_diagnostics = summarize_target_distributions(
        train_relevance, criterion
    )
    with open(
        os.path.join(output_dir, 'target_distribution_diagnostics.json'),
        'w',
        encoding='utf-8',
    ) as f:
        json.dump(target_diagnostics, f, indent=4, ensure_ascii=False)
    
    # 10. 排序模型训练
    if is_train:
        if competition_5day:
            for epoch in range(config['num_epochs']):
                print(f"\n=== Epoch {epoch+1}/{config['num_epochs']} ===")
                train_loss, train_metrics = train_ranking_model(
                    model, train_loader, criterion, optimizer, device, epoch, writer,
                    auxiliary_criterion=auxiliary_criterion,
                    auxiliary_weight=auxiliary_weight,
                )
                print(f"Train Loss: {train_loss:.4f}")
                for key, value in train_metrics.items():
                    print(f"Train {key}: {value:.8f}")
                scheduler.step()
                if writer:
                    writer.add_scalar(
                        'train/learning_rate', scheduler.get_last_lr()[0], global_step=epoch
                    )
                print("比赛协议：无验证集，不执行早停或epoch选择")

            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            test_loss, test_metrics = evaluate_ranking_model(
                model, val_loader, criterion, device, writer, config['num_epochs'] - 1
            )
            evaluation = {
                'protocol': 'single_signal_5day_competition_score',
                'epoch': int(config['num_epochs']),
                'signal_date': split['signal']['date'],
                'score_start': split['test']['start'],
                'score_end': split['test']['end'],
                'equal_weight_per_stock': 0.2,
                'test_loss': float(test_loss),
                'metrics': {key: float(value) for key, value in test_metrics.items()},
            }
            with open(
                os.path.join(output_dir, 'evaluation_competition_5day.json'),
                'w', encoding='utf-8',
            ) as handle:
                json.dump(evaluation, handle, indent=2, ensure_ascii=False)
            with open(
                os.path.join(output_dir, 'final_score.txt'), 'w', encoding='utf-8'
            ) as handle:
                handle.write(
                    f"Fixed epoch: {config['num_epochs']}\n"
                    f"Signal date: {split['signal']['date']}\n"
                    f"Score window: {split['test']['start']} ~ {split['test']['end']}\n"
                    f"Top-5 equal-weight return: {test_metrics.get('top5_return', 0.0):.8f}\n"
                )
            if writer:
                writer.close()
            return test_metrics.get('top5_return', 0.0)

        best_score = -float('inf')
        best_epoch = -1
        patience = config.get('early_stopping_patience', 10)
        no_improve_count = 0
        fixed_refit = config.get('fixed_refit', False)
        
        for epoch in range(config['num_epochs']):
            print(f"\n=== Epoch {epoch+1}/{config['num_epochs']} ===")
            
            # 训练
            train_loss, train_metrics = train_ranking_model(
                model, train_loader, criterion, optimizer, device, epoch, writer,
                auxiliary_criterion=auxiliary_criterion,
                auxiliary_weight=auxiliary_weight,
            )
            
            print(f"Train Loss: {train_loss:.4f}")
            for k, v in train_metrics.items():
                print(f"Train {k}: {v:.8f}")
            
            # 固定epoch重训时，外层测试只能在最后一轮结束后评估一次。
            # 中间epoch既不构造模型选择信号，也不保存“最佳”外层权重。
            if fixed_refit and epoch + 1 < config['num_epochs']:
                scheduler.step()
                if writer:
                    writer.add_scalar(
                        'train/learning_rate',
                        scheduler.get_last_lr()[0],
                        global_step=epoch,
                    )
                print("固定重训模式：本轮不访问外层评估集")
                continue

            # 验证；普通模式每轮执行，固定重训模式仅在最后一轮执行一次。
            eval_loss, eval_metrics = evaluate_ranking_model(
                model, val_loader, criterion, device, writer, epoch
            )
            
            print(f"Eval Loss: {eval_loss:.4f}")
            for k, v in eval_metrics.items():
                print(f"Eval {k}: {v:.8f}")
            
            # 学习率调度
            scheduler.step()
            if writer:
                writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step=epoch)
            

            # 以等权Top-5相对股票池的超额收益保存模型并早停。
            selection_metric = config.get(
                'model_selection_metric', 'top5_return'
            )
            current_score = eval_metrics.get(selection_metric, -float('inf'))
            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch + 1
                no_improve_count = 0
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                with open(
                    os.path.join(output_dir, 'evaluation_validation.json'),
                    'w',
                    encoding='utf-8',
                ) as f:
                    json.dump(
                        {
                            'epoch': best_epoch,
                            'selection_metric': selection_metric,
                            'metrics': {
                                key: float(value)
                                for key, value in eval_metrics.items()
                            },
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                print(
                    f"保存最佳模型 - val {selection_metric}: "
                    f"{best_score:.4%}"
                )
            else:
                no_improve_count += 1
                print(f"无提升 ({no_improve_count}/{patience})")
                if no_improve_count >= patience:
                    print(f"\n早停触发：连续 {patience} 轮无提升，停止训练")
                    break
        
        selection_metric = config.get(
            'model_selection_metric', 'top5_return'
        )
        print(
            f"\n训练完成！最佳 epoch: {best_epoch}, "
            f"最佳 val {selection_metric}: {best_score:.4%}"
        )
        test_metrics = None
        refit_performed = False
        if test_loader is not None and config.get('refit_after_validation', False):
            selection_checkpoint = os.path.join(output_dir, 'best_model.pth')
            shutil.copy2(
                selection_checkpoint,
                os.path.join(output_dir, 'validation_best_model.pth'),
            )

            refit_rows = (
                (processed_unscaled['日期'] >= train_start)
                & (processed_unscaled['日期'] <= val_end)
            )
            if 'is_member' in processed_unscaled.columns:
                refit_rows &= processed_unscaled['is_member'].fillna(False).astype(bool)
            if 'is_tradable' in processed_unscaled.columns:
                refit_rows &= processed_unscaled['is_tradable'].fillna(False).astype(bool)
            if 'in_signal_range' in processed_unscaled.columns:
                refit_rows &= processed_unscaled['in_signal_range'].fillna(False).astype(bool)
            refit_scaler = StandardScaler()
            refit_scaler.fit(processed_unscaled.loc[refit_rows, features])
            refit_processed = processed_unscaled.copy()
            refit_processed[features] = refit_scaler.transform(
                refit_processed[features]
            )
            joblib.dump(refit_scaler, os.path.join(output_dir, 'scaler.pkl'))

            refit_result = create_ranking_dataset_vectorized(
                refit_processed,
                features,
                config['sequence_length'],
                min_window_end_date=train_start,
                max_window_end_date=val_end,
                return_dates=True,
            )
            (
                refit_sequences, refit_targets, refit_relevance,
                refit_stock_indices, _,
            ) = refit_result
            refit_dataset = RankingDataset(
                refit_sequences, refit_targets, refit_relevance,
                refit_stock_indices,
            )
            refit_loader = DataLoader(
                refit_dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=False,
            )
            refit_test_sequences, refit_test_targets, refit_test_relevance, refit_test_stock_indices = (
                create_ranking_dataset_vectorized(
                    refit_processed,
                    features,
                    config['sequence_length'],
                    min_window_end_date=pd.to_datetime(split['test']['start']),
                    max_window_end_date=pd.to_datetime(split['test']['end']),
                )
            )
            refit_test_loader = DataLoader(
                RankingDataset(
                    refit_test_sequences, refit_test_targets,
                    refit_test_relevance, refit_test_stock_indices,
                ),
                batch_size=config['batch_size'],
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=False,
            )

            set_seed(config.get('seed', 42))
            model = INFO_TCN_iTransformer(
                input_dim=len(features), config=config, num_stocks=num_stocks
            ).to(device)
            if config.get('soft_gate_enabled', False):
                gate_parameters = list(model.market_gate.parameters())
                if not config.get('soft_gate_rule_mode', False):
                    gate_parameters.append(model.soft_gate_bias)
                gate_ids = {id(parameter) for parameter in gate_parameters}
                base_parameters = [
                    parameter for parameter in model.parameters()
                    if id(parameter) not in gate_ids
                ]
                optimizer = torch.optim.AdamW(
                    [
                        {'params': base_parameters, 'lr': config['learning_rate']},
                        {
                            'params': gate_parameters,
                            'lr': config['learning_rate'] * float(
                                config.get('soft_gate_learning_rate_multiplier', 20.0)
                            ),
                        },
                    ],
                    weight_decay=1e-5,
                )
            else:
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=config['learning_rate'],
                    weight_decay=1e-5,
                )
            scheduler = build_linear_scheduler(optimizer, config)
            print(
                f"开始完整训练区间重训: {len(refit_sequences)}日, "
                f"固定{best_epoch} epochs"
            )
            for refit_epoch in range(best_epoch):
                refit_loss, refit_metrics = train_ranking_model(
                    model, refit_loader, criterion, optimizer, device,
                    refit_epoch, writer,
                    auxiliary_criterion=auxiliary_criterion,
                    auxiliary_weight=auxiliary_weight,
                )
                print(
                    f"Refit Epoch {refit_epoch + 1}/{best_epoch} "
                    f"Loss: {refit_loss:.4f}"
                )
                scheduler.step()
            torch.save(model.state_dict(), selection_checkpoint)
            config['refit_after_validation'] = True
            config['refit_epoch'] = int(best_epoch)
            config['refit_train_start'] = split['train']['start']
            config['refit_train_end'] = split['validation']['end']
            with open(
                os.path.join(output_dir, 'config.json'), 'w', encoding='utf-8'
            ) as handle:
                json.dump(
                    {key: value for key, value in config.items() if not key.startswith('_')},
                    handle, indent=4, ensure_ascii=False, default=str,
                )
            test_loss, test_metrics = evaluate_ranking_model(
                model, refit_test_loader, criterion, device, writer, best_epoch
            )
            refit_performed = True
        elif test_loader is not None:
            checkpoint_path = os.path.join(output_dir, 'best_model.pth')
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            test_loss, test_metrics = evaluate_ranking_model(
                model, test_loader, criterion, device, writer, best_epoch
            )

        if test_metrics is not None:
            evaluation_test = {
                'protocol': (
                    'held_out_test_after_full_train_refit'
                    if refit_performed
                    else 'held_out_test_after_validation_selection'
                ),
                'best_epoch': int(best_epoch),
                'selection_metric': selection_metric,
                'refit_performed': refit_performed,
                'refit_train_start': (
                    split['train']['start'] if refit_performed else None
                ),
                'refit_train_end': (
                    split['validation']['end'] if refit_performed else None
                ),
                'test_start': split['test']['start'],
                'test_end': split['test']['end'],
                'test_loss': float(test_loss),
                'metrics': {
                    key: float(value) for key, value in test_metrics.items()
                },
            }
            with open(
                os.path.join(output_dir, 'evaluation_test.json'),
                'w', encoding='utf-8',
            ) as handle:
                json.dump(evaluation_test, handle, indent=2, ensure_ascii=False)
            print(f"Test Loss: {test_loss:.4f}")
            for key, value in test_metrics.items():
                print(f"Test {key}: {value:.8f}")
        with open(
            os.path.join(output_dir, 'final_score.txt'), 'w', encoding='utf-8'
        ) as f:
            f.write(
                f"Best epoch: {best_epoch}\n"
                f"Selection metric: {selection_metric}\n"
                f"Best validation score: {best_score:.8f}\n"
            )
            if test_metrics is not None:
                f.write(
                    f"Test Top-5 return: "
                    f"{test_metrics.get('top5_return', 0.0):.8f}\n"
                    f"Test Mean IC: {test_metrics.get('mean_ic', 0.0):.8f}\n"
                )

        if writer:
            writer.close()

        return best_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练 INFO-TCN-iTransformer')
    parser.add_argument('--seed', type=int, default=config.get('seed', 42))
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument(
        '--data-file',
        type=str,
        default=None,
        help='显式训练数据文件；比赛阶段可传入无PIT状态列的stock_data.csv',
    )
    parser.add_argument(
        '--init-checkpoint',
        type=str,
        default=None,
        help='从指定checkpoint继续训练，用于近期30日低学习率微调',
    )
    parser.add_argument(
        '--reuse-scaler-file',
        type=str,
        default=None,
        help='复用长期训练Scaler，避免近期微调重新拟合输入尺度',
    )
    parser.add_argument(
        '--reuse-soft-gate-rule-file',
        type=str,
        default=None,
        help='复用已冻结规则门控的原始市场均值、标准差和q70/q90阈值',
    )
    parser.add_argument(
        '--model-variant',
        choices=['hybrid', 'tcn_only', 'itransformer_only'],
        default=None,
    )
    parser.add_argument(
        '--feature-num',
        choices=[
            'baseline24',
            'baseline24_market4',
            'baseline24_market4_cross4',
            'baseline24_market4_component2',
        ],
        default=None,
        help='选择24维基线、28维市场特征组或32维市场×个股交叉组',
    )
    parser.add_argument(
        '--soft-gate',
        choices=['on', 'off'],
        default=None,
        help='显式启用或关闭市场状态软门控，便于严格复现实验',
    )
    parser.add_argument('--num-epochs', type=int, default=None)
    parser.add_argument('--scheduler-total-epochs', type=int, default=None)
    parser.add_argument('--early-stopping-patience', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--tcn-num-layers', type=int, default=None)
    parser.add_argument('--tcn-channels', type=int, default=None)
    parser.add_argument('--it-num-layers', type=int, default=None)
    parser.add_argument(
        '--ranking-loss-type',
        choices=[
            'head_focused',
            'weighted_ranknet',
            'weighted_ranknet_stability',
        ],
        default=None,
    )
    parser.add_argument(
        '--listwise-target-type',
        choices=[
            'normalized_rank_softmax',
            'percentile_target',
            'ordinal_softmax',
        ],
        default=None,
    )
    parser.add_argument('--listwise-temperature', type=float, default=None)
    parser.add_argument('--enable-info', action='store_true')
    parser.add_argument(
        '--fixed-refit',
        action='store_true',
        help='固定epoch重训；仅在最后一轮评估一次validation（其角色为外层测试）',
    )
    parser.add_argument(
        '--refit-after-validation',
        action='store_true',
        help='用validation选出的epoch在train+validation完整区间从头重训后评估test',
    )
    parser.add_argument('--auxiliary-loss-weight', type=float, default=None)
    parser.add_argument(
        '--global-rank-weight',
        type=float,
        default=None,
        help='head_focused损失中的全横截面Pairwise正则权重',
    )
    parser.add_argument(
        '--stability-weight',
        type=float,
        default=None,
        help='head_focused损失中的RankIC稳定性正则权重',
    )
    parser.add_argument(
        '--shortcut-correlation-weight',
        type=float,
        default=None,
        help='模型分数与波动率指数绝对相关系数的惩罚权重',
    )
    parser.add_argument('--train-days', type=int, default=None)
    parser.add_argument('--gap-days', type=int, default=None)
    parser.add_argument(
        '--gap2-days', type=int, default=None,
        help='validation与test之间的独立交易日隔离长度；默认等于gap-days',
    )
    parser.add_argument('--val-days', type=int, default=None)
    parser.add_argument('--test-days', type=int, default=None)
    parser.add_argument('--data-lookback-years', type=int, default=None)
    parser.add_argument(
        '--competition-train-days',
        type=int,
        default=None,
        help='competition_5day滚动窗口中的训练目标交易日数',
    )
    parser.add_argument(
        '--competition-train-end-date',
        type=str,
        default=None,
        help='competition_5day训练目标的显式截止日期；与competition-train-days互斥',
    )
    parser.add_argument(
        '--window-end-date',
        type=str,
        default=None,
        help='滚动窗口最后一个5日评分交易日；仅切分目标日期，不截断因果历史特征',
    )
    parser.add_argument(
        '--split-start-date',
        type=str,
        default=None,
        help='仅限制train/validation目标日期起点；更早数据仍可作为60日序列上下文',
    )
    parser.add_argument(
        '--sampling-strategy',
        choices=['natural', 'stratified_regime'],
        default=None,
        help='训练日期采样；验证和测试始终保持自然分布',
    )
    parser.add_argument('--weak-target-share', type=float, default=None)
    parser.add_argument(
        '--training-protocol',
        choices=['competition_5day', 'rolling_validation'],
        default=None,
        help='competition_5day不使用验证集；rolling_validation保留论文研究三段切分',
    )
    args = parser.parse_args()
    config['seed'] = args.seed
    if args.output_dir:
        config['output_dir'] = os.path.abspath(args.output_dir)
    if args.data_file:
        config['data_file'] = os.path.abspath(args.data_file)
    if args.sampling_strategy:
        config['sampling_strategy'] = args.sampling_strategy
    if args.training_protocol:
        config['training_protocol'] = args.training_protocol
    if args.competition_train_days is not None:
        if args.competition_train_days <= 0:
            raise ValueError('competition-train-days必须为正数')
        config['competition_train_days'] = args.competition_train_days
    if args.competition_train_end_date:
        config['competition_train_end_date'] = args.competition_train_end_date
    if args.window_end_date:
        config['window_end_date'] = args.window_end_date
    if args.weak_target_share is not None:
        if abs(args.weak_target_share - 0.30) > 1e-12:
            raise ValueError('当前预注册实验只授权weak_target_share=0.30')
        config['weak_target_share'] = args.weak_target_share
    if args.init_checkpoint:
        config['init_checkpoint'] = os.path.abspath(args.init_checkpoint)
    if args.reuse_scaler_file:
        config['reuse_scaler_file'] = os.path.abspath(args.reuse_scaler_file)
    if args.reuse_soft_gate_rule_file:
        config['reuse_soft_gate_rule_file'] = os.path.abspath(
            args.reuse_soft_gate_rule_file
        )
    if args.model_variant:
        config['model_variant'] = args.model_variant
    if args.feature_num:
        config['feature_num'] = args.feature_num
        config['include_features'] = list(
            BASELINE24_FEATURES
            + (
                MARKET4_FEATURES
                if args.feature_num in {
                    'baseline24_market4',
                    'baseline24_market4_cross4',
                    'baseline24_market4_component2',
                }
                else []
            )
            + (CROSS4_FEATURES if args.feature_num == 'baseline24_market4_cross4' else [])
            + (
                COMPONENT2_FEATURES
                if args.feature_num == 'baseline24_market4_component2'
                else []
            )
        )
    if args.soft_gate:
        config['soft_gate_enabled'] = args.soft_gate == 'on'
    if args.num_epochs is not None:
        config['num_epochs'] = args.num_epochs
    if args.scheduler_total_epochs is not None:
        if args.scheduler_total_epochs <= 0:
            raise ValueError('scheduler-total-epochs必须为正数')
        config['scheduler_total_epochs'] = args.scheduler_total_epochs
    if args.early_stopping_patience is not None:
        config['early_stopping_patience'] = args.early_stopping_patience
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if args.tcn_num_layers is not None:
        config['tcn_num_layers'] = args.tcn_num_layers
    if args.tcn_channels is not None:
        config['tcn_channels'] = args.tcn_channels
    if args.it_num_layers is not None:
        config['it_num_layers'] = args.it_num_layers
    if args.ranking_loss_type is not None:
        config['ranking_loss_type'] = args.ranking_loss_type
    if args.listwise_target_type is not None:
        config['listwise_target_type'] = args.listwise_target_type
    if args.listwise_temperature is not None:
        if args.listwise_temperature <= 0:
            raise ValueError('listwise-temperature必须为正数')
        config['listwise_temperature'] = args.listwise_temperature
    if args.enable_info:
        config['use_info'] = True
    if args.fixed_refit:
        config['fixed_refit'] = True
    if args.refit_after_validation:
        config['refit_after_validation'] = True
    if args.auxiliary_loss_weight is not None:
        if args.auxiliary_loss_weight < 0:
            raise ValueError('auxiliary-loss-weight不能为负数')
        config['auxiliary_loss_weight'] = args.auxiliary_loss_weight
    if args.global_rank_weight is not None:
        if args.global_rank_weight < 0:
            raise ValueError('global-rank-weight不能为负数')
        config['global_rank_weight'] = args.global_rank_weight
    if args.stability_weight is not None:
        if args.stability_weight < 0:
            raise ValueError('stability-weight不能为负数')
        apply_stability_weight_override(config, args.stability_weight)
    if args.shortcut_correlation_weight is not None:
        if args.shortcut_correlation_weight < 0:
            raise ValueError('shortcut-correlation-weight不能为负数')
        config['shortcut_correlation_weight'] = (
            args.shortcut_correlation_weight
        )
    if args.data_lookback_years is not None:
        config['data_lookback_years'] = (
            None if args.data_lookback_years == 0 else args.data_lookback_years
        )
    if args.split_start_date is not None:
        config['split_start_date'] = args.split_start_date
    explicit_split_days = False
    for arg_name, config_name in (
        ('train_days', 'train_days'),
        ('gap_days', 'gap_days'),
        ('gap2_days', 'gap2_days'),
        ('val_days', 'val_days'),
        ('test_days', 'test_days'),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            config[config_name] = value
            if arg_name in {'train_days', 'val_days', 'test_days'}:
                explicit_split_days = True
    config['_explicit_split_days'] = explicit_split_days

    # 多进程保护
    mp.set_start_method('spawn', force=True)
    best_score = main()
    if config.get('training_protocol') == 'competition_5day':
        print(f"\n########## 单次5日比赛得分: {best_score:.4%} ##########")
    else:
        print(
            "\n########## 训练完成！最佳 val "
            f"{config.get('model_selection_metric', 'top5_return')}: "
            f"{best_score:.4%} ##########"
        )
