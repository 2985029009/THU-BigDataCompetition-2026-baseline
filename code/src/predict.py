import os
import json
import multiprocessing as mp
import argparse

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import config
from model import INFO_TCN_iTransformer
from utils import (
	BASELINE24_FEATURES,
	COMPONENT2_FEATURES,
	CROSS4_FEATURES,
	MARKET4_FEATURES,
	add_causal_market_features,
	add_market_component_cross_features,
	add_market_cross_features,
	engineer_features_baseline24,
)
from pit_data import point_in_time_cross_sectional_rank, sha256_file


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


def preprocess_predict_data(df, stockid2idx):
	assert config['feature_num'] in feature_engineer_func_map, f"Unsupported feature_num: {config['feature_num']}"
	feature_engineer = feature_engineer_func_map[config['feature_num']]
	feature_columns = feature_cloums_map[config['feature_num']]

	df = df.copy()
	df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
	if config['feature_num'] in {
		'baseline24_market4',
		'baseline24_market4_cross4',
		'baseline24_market4_component2',
	}:
		df = add_causal_market_features(df)
	groups = [group for _, group in df.groupby('股票代码', sort=False)]
	if len(groups) == 0:
		raise ValueError('输入数据为空，无法预测')

	num_processes = min(10, mp.cpu_count())
	print('cpus!!!!!!!!!!!!!!!!!!',mp.cpu_count())
	with mp.Pool(processes=num_processes) as pool:
		processed_list = list(tqdm(pool.imap(feature_engineer, groups), total=len(groups), desc='预测集特征工程'))

	processed = pd.concat(processed_list).reset_index(drop=True)
	processed['instrument'] = processed['股票代码'].map(stockid2idx)
	processed = processed.dropna(subset=['instrument']).copy()
	processed['instrument'] = processed['instrument'].astype(np.int64)
	processed['日期'] = pd.to_datetime(processed['日期'])

	return processed, feature_columns


def build_inference_sequences(data, features, sequence_length, stock_ids, latest_date):
	sequences, sequence_stock_ids = [], []
	for stock_id in stock_ids:
		stock_history = data[
			(data['股票代码'] == stock_id) &
			(data['日期'] <= latest_date)
		].sort_values('日期').tail(sequence_length)

		if len(stock_history) == sequence_length:
			sequences.append(stock_history[features].values.astype(np.float32))
			sequence_stock_ids.append(stock_id)

	if len(sequences) == 0:
		raise ValueError('没有可用于预测的股票序列，请检查数据与 sequence_length')

	return np.asarray(sequences, dtype=np.float32), sequence_stock_ids


def prepare_predict_features(processed, features, scaler):
	"""Apply the exact feature transforms used by prediction before inference.

	This function deliberately does not construct labels.  It is shared by the
	prediction CLI and offline evaluation so that the evaluated universe is the
	same universe that would be sent to the deployed model.
	"""
	processed = processed.copy()
	base_features = config.get('base_feature_names', features)
	if config.get('feature_transform') == 'cross_sectional_rank':
		if 'is_member' in processed.columns:
			processed = point_in_time_cross_sectional_rank(
				processed, base_features
			)
		else:
			processed[base_features] = (
				processed.groupby('日期', sort=False)[base_features]
				.rank(method='average', pct=True)
			)
	if config.get('feature_num') == 'baseline24_market4_cross4':
		center = config.get('market_cross_center_raw')
		scale = config.get('market_cross_scale_raw')
		if center is None or scale is None:
			raise ValueError('交叉特征模型缺少训练期市场压力统计量')
		processed = add_market_cross_features(processed, center, scale)
	elif config.get('feature_num') == 'baseline24_market4_component2':
		center = config.get('market_cross_center_raw')
		scale = config.get('market_cross_scale_raw')
		if center is None or scale is None:
			raise ValueError('市场分量交叉模型缺少训练期标准化统计量')
		processed = add_market_component_cross_features(processed, center, scale)
	processed[features] = processed[features].replace([np.inf, -np.inf], np.nan)
	processed = processed.dropna(subset=features).copy()
	processed[features] = scaler.transform(processed[features])
	return processed


def get_predict_candidate_ids(processed, prediction_date):
	"""Return the point-in-time candidate universe used by ``predict.py``."""
	prediction_date = pd.Timestamp(prediction_date).normalize()
	if {'is_member', 'is_tradable'}.issubset(processed.columns):
		candidates = processed[
			processed['日期'].eq(prediction_date)
			& processed['is_member'].astype(bool)
			& processed['is_tradable'].astype(bool)
		]['股票代码'].unique()
	else:
		candidates = processed[
			processed['日期'].eq(prediction_date)
		]['股票代码'].unique()
	return sorted(candidates)


def main():
	# 使用项目根目录的 output 文件夹
	project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	parser = argparse.ArgumentParser(description='生成前5只股票预测')
	parser.add_argument('--model-dir', default=config['output_dir'])
	parser.add_argument(
		'--data-file',
		default=None,
		help='显式推理数据优先于训练配置中的本地数据路径',
	)
	parser.add_argument(
		'--prediction-date',
		default=None,
		help='指定历史信号日（YYYY-MM-DD）；默认使用数据中的最新日期',
	)
	parser.add_argument('--output', default=os.path.join(project_root, 'output', 'result.csv'))
	parser.add_argument(
		'--soft-gate-override',
		type=float,
		default=None,
		help='Counterfactual inference only: force the final soft gate to [0, 1]',
	)
	parser.add_argument(
		'--full-ranking-output',
		default=None,
		help='可选：保存完整股票池的原始模型分数与排名，用于离线排序诊断',
	)
	args = parser.parse_args()

	model_dir = os.path.abspath(args.model_dir)
	explicit_data_file = os.path.abspath(args.data_file) if args.data_file else None
	data_file = explicit_data_file or os.path.abspath(
		config.get('data_file') or os.path.join(project_root, 'data', 'train.csv')
	)
	output_path = os.path.abspath(args.output)
	model_path = os.path.join(model_dir, 'best_model.pth')
	scaler_path = os.path.join(model_dir, 'scaler.pkl')

	if not os.path.exists(model_path):
		raise FileNotFoundError(f'未找到模型文件: {model_path}')
	if not os.path.exists(scaler_path):
		raise FileNotFoundError(f'未找到Scaler文件: {scaler_path}')

	# 从模型目录的 config.json 读取实际训练架构（INFO 搜索可能改变了架构参数），
	# 避免使用 config.py 的旧参数构建模型导致权重加载失败。
	cfg_path = os.path.join(model_dir, 'config.json')
	if os.path.exists(cfg_path):
		with open(cfg_path, 'r', encoding='utf-8') as f:
			trained_config = json.load(f)
			# 模型目录配置必须是完整快照；清除当前代码新增字段，避免旧模型
			# 在推理时意外继承尚未训练过的新门控规则。
			config.clear()
			config.update(trained_config)
		trained_data = config.get('dataset_file')
		if explicit_data_file:
			data_file = explicit_data_file
		elif trained_data:
			data_file = trained_data
			expected_hash = config.get('dataset_sha256')
			if expected_hash and sha256_file(data_file) != expected_hash:
				raise ValueError('预测数据哈希与训练配置不一致；如需使用新数据请显式确认并重新训练')

	raw_df = pd.read_csv(data_file, dtype={'股票代码': str})
	raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
	raw_df['日期'] = pd.to_datetime(raw_df['日期'])
	for boolean_column in ['is_member', 'is_suspended', 'is_tradable', 'in_signal_range']:
		if boolean_column in raw_df.columns:
			raw_df[boolean_column] = (
				raw_df[boolean_column].astype(str).str.lower().isin({'true', '1'})
			)
	if args.prediction_date:
		latest_date = pd.Timestamp(args.prediction_date).normalize()
		available_dates = raw_df['日期'].dt.normalize()
		if latest_date not in set(available_dates):
			raise ValueError(f'预测日期不在数据中: {latest_date.date()}')
	else:
		latest_date = raw_df['日期'].max()

	stock_ids = sorted(raw_df['股票代码'].unique())
	if args.soft_gate_override is not None:
		if not 0.0 <= args.soft_gate_override <= 1.0:
			raise ValueError('--soft-gate-override must be between 0 and 1')
		config['soft_gate_override'] = args.soft_gate_override

	stockid2idx = {sid: idx for idx, sid in enumerate(stock_ids)}

	processed, all_features = preprocess_predict_data(raw_df, stockid2idx)
	features = config.get('feature_names', all_features)
	scaler = joblib.load(scaler_path)
	processed = prepare_predict_features(processed, features, scaler)

	sequence_length = config['sequence_length']
	latest_candidates = get_predict_candidate_ids(processed, latest_date)
	sequences_np, sequence_stock_ids = build_inference_sequences(
		processed,
		features,
		sequence_length,
		latest_candidates,
		latest_date,
	)

	if torch.cuda.is_available():
		device = torch.device('cuda')
	elif torch.backends.mps.is_available():
		device = torch.device('mps')
	else:
		device = torch.device('cpu')

	model = INFO_TCN_iTransformer(input_dim=len(features), config=config, num_stocks=len(stock_ids))
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.to(device)
	model.eval()

	with torch.no_grad():
		x = torch.from_numpy(sequences_np).unsqueeze(0).to(device)  # [1, N, L, F]
		scores = model(x).squeeze(0).detach().cpu().numpy()         # [N]
	gate_values = getattr(model, 'last_gate_values', None)

	order = np.argsort(scores)[::-1]
	ranked_stock_ids = [sequence_stock_ids[i] for i in order]
	if args.full_ranking_output:
		full_ranking_path = os.path.abspath(args.full_ranking_output)
		full_ranking = pd.DataFrame({
			'stock_id': ranked_stock_ids,
			'raw_score': scores[order],
			'raw_rank': np.arange(1, len(order) + 1),
		})
		os.makedirs(os.path.dirname(full_ranking_path), exist_ok=True)
		full_ranking.to_csv(full_ranking_path, index=False)
		print(f'完整排序已写入: {full_ranking_path}')

	# 仅输出前5，权重固定 0.2
	if len(ranked_stock_ids) < 5:
		raise ValueError(f'可预测股票不足5只，当前仅有 {len(ranked_stock_ids)} 只')
	top5 = ranked_stock_ids[:5]
	output_df = pd.DataFrame({
		'stock_id': top5,
		'weight': [0.2] * len(top5),
	})
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	output_df.to_csv(output_path, index=False)

	print(f'预测日期: {latest_date.date()}')
	print(f'参与排序股票数: {len(ranked_stock_ids)}')
	if gate_values is not None:
		print(f'防御软门控权重: {float(gate_values.mean().cpu()):.6f}')
	print(f'结果已写入: {output_path}')


if __name__ == '__main__':
	mp.set_start_method('spawn', force=True)
	main()
