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
from predict import (
    build_inference_sequences,
    get_predict_candidate_ids,
    prepare_predict_features,
    preprocess_predict_data,
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
       