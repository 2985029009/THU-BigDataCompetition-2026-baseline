"""
模型评测脚本：横截面 RankIC 评测（量化/金融的权威口径）

与训练/预测一致，每个交易日把当天全部股票一起输入模型 [1, N, 60, F]，
得到 N 个分数后与未来5日收益做 Spearman RankIC，再按日聚合。

默认在 data_split.json 的 test 段上评测（严格隔离，非 in-sample）。
可用 --split {test,val,train,full} 切换评测区间。
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import joblib
from scipy.stats import spearmanr

# 添加 src 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code', 'src'))

from config import config
from model import INFO_TCN_iTransformer
from utils import (
    BASELINE24_FEATURES,
    add_causal_market_features,
    engineer_features_baseline24,
    create_ranking_dataset_vectorized,
)
from pit_data import (
    apply_contest_score_label,
    point_in_time_cross_sectional_rank,
    sha256_file,
)

# 特征列（与train.py保持一致）
feature_columns = list(BASELINE24_FEATURES)


def load_and_preprocess(data_path, scaler_path, selected_features, feature_num):
    """加载数据 + 全时间线特征工程 + 标签 + 用训练期拟合的 scaler 变换。"""
    print("加载数据 + 特征工程...")
    df = pd.read_csv(data_path, dtype={'股票代码': str})
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期'] = pd.to_datetime(df['日期'])
    for boolean_column in ['is_member', 'is_suspended', 'is_tradable', 'in_signal_range']:
        if boolean_column in df.columns:
            df[boolean_column] = (
                df[boolean_column].astype(str).str.lower().isin({'true', '1'})
            )
    if config.get('label_mode') == 'contest_score':
        df = apply_contest_score_label(df)

    stock_ids = sorted(df['股票代码'].unique())
    stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

    audit_columns = [
        column for column in [
            'label', 'is_member', 'is_suspended', 'is_tradable',
            'in_signal_range', 'label_entry_failed', 'adj_factor', 'source',
        ] if column in df.columns
    ]
    audit_frame = df[['股票代码', '日期'] + audit_columns].copy()
    if feature_num == 'baseline24_market4':
        df = add_causal_market_features(df)
    groups = [g for _, g in df.groupby('股票代码', sort=False)]
    if feature_num not in {'baseline24', 'baseline24_market4'}:
        raise ValueError(f"评估脚本不支持feature_num={feature_num}")
    processed = pd.concat(
        [engineer_features_baseline24(g) for g in groups]
    ).reset_index(drop=True)
    if audit_columns:
        processed = processed.drop(columns=audit_columns, errors='ignore').merge(
            audit_frame,
            on=['股票代码', '日期'],
            how='left',
            validate='one_to_one',
        )
    processed['instrument'] = processed['股票代码'].map(stockid2idx)
    processed = processed.dropna(subset=['instrument']).copy()
    processed['instrument'] = processed['instrument'].astype(np.int64)

    # PIT数据已经按统一交易日历生成可执行标签；旧数据保留历史兼容逻辑。
    processed = processed.sort_values(['股票代码', '日期']).reset_index(drop=True)
    if 'label' not in processed.columns:
        processed['open_t1'] = processed.groupby('股票代码')['开盘'].shift(-1)
        processed['open_t5'] = processed.groupby('股票代码')['开盘'].shift(-5)
        processed['label'] = (
            (processed['open_t5'] - processed['open_t1'])
            / (processed['open_t1'] + 1e-12)
        )
        processed.drop(columns=['open_t1', 'open_t5'], inplace=True)
    else:
        processed['label'] = pd.to_numeric(processed['label'], errors='coerce')

    base_features = config.get('base_feature_names', selected_features)
    if config.get('feature_transform') == 'cross_sectional_rank':
        if 'is_member' in processed.columns:
            processed = point_in_time_cross_sectional_rank(processed, base_features)
        else:
            processed[base_features] = (
                processed.groupby('日期', sort=False)[base_features]
                .rank(method='average', pct=True)
            )
    # 标准化：使用训练阶段保存的 scaler（仅在 train 段拟合）
    scaler = joblib.load(scaler_path)
    processed[selected_features] = processed[selected_features].replace(
        [np.inf, -np.inf], np.nan
    )
    processed = processed.dropna(subset=selected_features).copy()
    processed[selected_features] = scaler.transform(processed[selected_features])

    return processed, len(stock_ids)


def evaluate_cross_section(model, processed, selected_features, device, min_date=None, max_date=None):
    """逐日评估Top-K组合结果，并保留RankIC作为辅助诊断。"""
    print("构建每日横截面样本...")
    seqs, targets, _, _, sample_dates = create_ranking_dataset_vectorized(
        processed, selected_features, config['sequence_length'],
        min_window_end_date=min_date, max_window_end_date=max_date,
        return_dates=True,
    )

    print("逐日Top-K组合评估 + RankIC辅助诊断 ...")
    ics = []
    daily_results = []
    with torch.no_grad():
        for sample_date, day_seq, day_tgt in zip(sample_dates, seqs, targets):
            if len(day_seq) < 10:
                continue
            x = torch.FloatTensor(np.asarray(day_seq)).unsqueeze(0).to(device)  # [1, N, L, F]
            scores = model(x).squeeze(0).cpu().numpy()  # [N]
            day_tgt = np.asarray(day_tgt)
            ic, _ = spearmanr(scores, day_tgt)
            if np.isnan(ic):
                ic = 0.0
            ics.append(float(ic))
            row = {
                'date': str(pd.Timestamp(sample_date).date()),
                'universe_return': float(day_tgt.mean()),
                'rank_ic': float(ic),
            }
            for k in (5, 10, 20):
                actual_k = min(k, len(scores))
                selected = np.argsort(scores)[-actual_k:]
                portfolio_return = float(day_tgt[selected].mean())
                row[f'top{k}_return'] = portfolio_return
                row[f'top{k}_excess_return'] = (
                    portfolio_return - row['universe_return']
                )
            true_order = np.argsort(day_tgt)
            percentiles = np.empty(len(day_tgt), dtype=np.float64)
            percentiles[true_order] = np.linspace(0.0, 1.0, len(day_tgt))
            pred_top5 = np.argsort(scores)[-5:][::-1]
            ideal_top5 = np.argsort(percentiles)[-5:][::-1]
            discounts = 1.0 / np.log2(np.arange(2, 7))
            row['ndcg_at_5'] = float(
                np.sum(percentiles[pred_top5] * discounts)
                / max(np.sum(percentiles[ideal_top5] * discounts), 1e-12)
            )
            row['top5_true_percentile'] = float(
                percentiles[pred_top5].mean()
            )
            daily_results.append(row)

    ics = np.array(ics)
    if not daily_results:
        return {'n_dates': 0, 'mean_ic': 0.0, 'std_ic': 0.0, 'icir': 0.0,
                'positive_ratio': 0.0, 'top5_realized_mean': 0.0,
                'daily_results': []}
    mean_ic = float(ics.mean())
    std_ic = float(ics.std())
    metrics = {
        'n_dates': int(len(daily_results)),
        'mean_ic': mean_ic,
        'std_ic': std_ic,
        'icir': float(mean_ic / std_ic) if std_ic > 0 else 0.0,
        'positive_ratio': float((ics > 0).mean()),
        'daily_ic': ics.tolist(),
        'daily_results': daily_results,
    }
    for k in (5, 10, 20):
        metrics[f'top{k}_return_mean'] = float(np.mean([
            row[f'top{k}_return'] for row in daily_results
        ]))
        metrics[f'top{k}_excess_return_mean'] = float(np.mean([
            row[f'top{k}_excess_return'] for row in daily_results
        ]))
    metrics['top5_realized_mean'] = metrics['top5_return_mean']
    metrics['worst_daily_top5_excess_return'] = float(min(
        row['top5_excess_return'] for row in daily_results
    ))
    metrics['positive_top5_excess_day_ratio'] = float(np.mean([
        row['top5_excess_return'] > 0 for row in daily_results
    ]))
    metrics['ndcg_at_5_mean'] = float(np.mean([
        row['ndcg_at_5'] for row in daily_results
    ]))
    metrics['top5_true_percentile_mean'] = float(np.mean([
        row['top5_true_percentile'] for row in daily_results
    ]))
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Top-K组合收益评测')
    parser.add_argument('--split', type=str, default='test',
                        choices=['test', 'validation', 'train', 'full'],
                        help='评测区间，默认 test（严格隔离）')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='模型目录，默认使用config中的output_dir')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = args.model_dir or os.path.abspath(config['output_dir'])
    model_path = os.path.join(model_dir, 'best_model.pth')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    data_path = config.get('dataset_file') or config.get('data_file')
    if not data_path:
        data_path = os.path.join(base_dir, 'data', 'train.csv')
    elif not os.path.isabs(data_path):
        data_path = os.path.join(base_dir, data_path)

    # 从模型目录读取实际训练架构（INFO 可能改变了架构参数）
    cfg_path = os.path.join(model_dir, 'config.json')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r', encoding='utf-8') as f:
            config.update(json.load(f))
        data_path = config.get('dataset_file') or config.get('data_file') or data_path
        if not os.path.isabs(data_path):
            data_path = os.path.join(base_dir, data_path)
        expected_hash = config.get('dataset_sha256')
        if expected_hash and sha256_file(data_path) != expected_hash:
            raise ValueError("评估数据哈希与训练配置不一致，拒绝评估")

    # 读取数据切分，确定评测区间
    split_path = os.path.join(model_dir, 'data_split.json')
    min_date = max_date = None
    if args.split != 'full' and os.path.exists(split_path):
        with open(split_path, 'r', encoding='utf-8') as f:
            split = json.load(f)
        if args.split == 'test':
            if split.get('split_method') == 'competition_single_signal_5day_score':
                # 比赛测试段是未来5个评分日；模型只在此前的单一信号日决策一次。
                min_date = max_date = split['signal']['date']
            else:
                min_date, max_date = split['test']['start'], split['test']['end']
        elif args.split == 'validation':
            min_date, max_date = split['validation']['start'], split['validation']['end']
        elif args.split == 'train':
            min_date, max_date = split['train']['start'], split['train']['end']
    elif args.split != 'full':
        print(f"警告: 未找到 {split_path}，回退为全历史 in-sample 评测")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Device: {device}")
    print(f"Model : {model_path}")
    print(f"Split : {args.split}  区间: {min_date} ~ {max_date}")
    print()

    selected_features = config.get('feature_names', feature_columns)
    processed, num_stocks = load_and_preprocess(
        data_path,
        scaler_path,
        selected_features,
        config['feature_num'],
    )

    token_count = len(selected_features)
    print(f"加载模型 (tcn_ch={config['tcn_channels']}, tcn_layers={config['tcn_num_layers']}, "
          f"it_layers={config['it_num_layers']}, tokens={token_count})...")
    model = INFO_TCN_iTransformer(input_dim=len(selected_features), config=config, num_stocks=num_stocks)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    metrics = evaluate_cross_section(
        model, processed, selected_features, device, min_date=min_date, max_date=max_date
    )

    print("\n" + "=" * 50)
    print(f"Top-K组合收益评测结果  (split={args.split})")
    print("=" * 50)
    print(f"  评测日期数 : {metrics['n_dates']}")
    print(f"  Top5平均收益: {metrics['top5_return_mean']:.4%}")
    print(f"  Top5平均超额: {metrics['top5_excess_return_mean']:.4%}")
    print(f"  Top10平均超额: {metrics['top10_excess_return_mean']:.4%}")
    print(f"  Top20平均超额: {metrics['top20_excess_return_mean']:.4%}")
    print(f"  NDCG@5     : {metrics['ndcg_at_5_mean']:.4f}")
    print(f"  Mean IC(辅助): {metrics['mean_ic']:.4f}")
    print(f"  Std IC     : {metrics['std_ic']:.4f}")
    print(f"  ICIR       : {metrics['icir']:.4f}")
    print(f"  IC>0 比例  : {metrics['positive_ratio']:.2%}")

    # 保存评测结果到模型目录
    out_path = os.path.join(model_dir, f'evaluation_{args.split}.json')
    save_metrics = {
        k: v for k, v in metrics.items()
        if k not in {'daily_ic', 'daily_results'}
    }
    save_metrics['split'] = args.split
    save_metrics['date_range'] = [min_date, max_date]
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(save_metrics, f, indent=2, ensure_ascii=False)
    daily_path = os.path.join(
        model_dir, f'evaluation_{args.split}_daily.csv'
    )
    pd.DataFrame(metrics.get('daily_results', [])).to_csv(
        daily_path, index=False, encoding='utf-8-sig'
    )
    print(f"\n结果已保存: {out_path}")
    print(f"逐日组合结果已保存: {daily_path}")


if __name__ == '__main__':
    main()
