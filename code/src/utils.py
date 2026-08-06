import pandas as pd
import numpy as np
import joblib
import os
from scipy.stats import rankdata
from tqdm import tqdm


BASELINE24_FEATURES = [
    '振幅', 'STD5', 'STD10', 'STD20', 'STD60',
    'volatility_10', 'volatility_20', 'atr_14',
    '成交额', '换手率', 'volume_ratio', 'volume_change',
    'VMA5', 'VMA20', 'VSTD20', 'WVMA20',
    'ROC5', 'ROC20', 'ROC60', 'RANK5', 'RANK20', 'RANK60',
    'RSV20', 'MA20',
]

MARKET4_FEATURES = [
    'market_return_20',
    'market_volatility_20',
    'market_drawdown_60',
    'market_breadth_20',
]

MARKET_STRESS_DIRECTIONS = np.asarray([-1.0, 1.0, -1.0, -1.0])

LOW_VOLATILITY_FEATURES = [
    '振幅', 'STD5', 'STD10', 'STD20', 'STD60',
    'volatility_10', 'volatility_20', 'atr_14',
]

CROSS4_FEATURES = [
    'stress_x_ROC60',
    'stress_x_low_vol',
    'stress_x_RANK20',
    'stress_x_volume_change',
]

COMPONENT2_FEATURES = [
    'volatility_stress_x_ROC20',
    'drawdown_stress_x_STD20',
]


def fit_market_stress_statistics(df, fit_rows, date_column='日期'):
    """仅用指定训练行拟合市场压力的时间序列统计量。"""
    missing = set([date_column, *MARKET4_FEATURES]).difference(df.columns)
    if missing:
        raise ValueError(f"市场压力拟合缺少字段: {sorted(missing)}")

    fit_rows = pd.Series(fit_rows, index=df.index).fillna(False).astype(bool)
    daily_market = (
        df.loc[fit_rows, [date_column, *MARKET4_FEATURES]]
        .groupby(date_column, sort=True)[MARKET4_FEATURES]
        .mean()
    )
    if daily_market.empty:
        raise ValueError('训练区间没有可用于拟合市场压力的日度样本')
    center = daily_market.mean(axis=0)
    scale = daily_market.std(axis=0, ddof=0)
    if not np.isfinite(center.to_numpy()).all():
        raise ValueError('市场压力中心包含非有限值')
    if not np.isfinite(scale.to_numpy()).all() or (scale <= 0).any():
        raise ValueError('市场压力尺度必须为正的有限值')
    return daily_market, center, scale


def add_market_cross_features(df, market_center, market_scale):
    """
    用已排名的个股特征和训练期市场统计量构造4个显式交叉项。

    market_center/market_scale 必须来自训练日并在验证、测试和推理时复用。
    """
    required = set(MARKET4_FEATURES + LOW_VOLATILITY_FEATURES + [
        'ROC60', 'RANK20', 'volume_change',
    ])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"交叉特征缺少字段: {sorted(missing)}")

    center = np.asarray(market_center, dtype=float)
    scale = np.asarray(market_scale, dtype=float)
    if center.shape != (len(MARKET4_FEATURES),):
        raise ValueError('市场压力中心维度不正确')
    if scale.shape != center.shape or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('市场压力尺度维度不正确或包含非正值')

    result = df.copy()
    market = result[MARKET4_FEATURES].apply(pd.to_numeric, errors='coerce')
    stress = (
        ((market.to_numpy(dtype=float) - center) / scale)
        * MARKET_STRESS_DIRECTIONS
    ).mean(axis=1)
    low_volatility_score = (
        1.0
        - result[LOW_VOLATILITY_FEATURES].apply(pd.to_numeric, errors='coerce')
    ).mean(axis=1)

    result['stress_x_ROC60'] = stress * pd.to_numeric(result['ROC60'], errors='coerce')
    result['stress_x_low_vol'] = stress * low_volatility_score
    result['stress_x_RANK20'] = stress * pd.to_numeric(result['RANK20'], errors='coerce')
    result['stress_x_volume_change'] = stress * pd.to_numeric(
        result['volume_change'], errors='coerce'
    )
    return result


def add_market_component_cross_features(df, market_center, market_scale):
    """构造两个仅由训练期统计量标准化的市场分量交叉项。"""
    required = set(MARKET4_FEATURES + ['ROC20', 'STD20'])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"市场分量交叉特征缺少字段: {sorted(missing)}")

    center = np.asarray(market_center, dtype=float)
    scale = np.asarray(market_scale, dtype=float)
    if center.shape != (len(MARKET4_FEATURES),):
        raise ValueError('市场分量交叉中心维度不正确')
    if scale.shape != center.shape or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('市场分量交叉尺度维度不正确或包含非正值')

    result = df.copy()
    market = result[MARKET4_FEATURES].apply(pd.to_numeric, errors='coerce')
    market_z = (market.to_numpy(dtype=float) - center) / scale
    volatility_index = MARKET4_FEATURES.index('market_volatility_20')
    drawdown_index = MARKET4_FEATURES.index('market_drawdown_60')
    volatility_stress = market_z[:, volatility_index]
    drawdown_stress = -market_z[:, drawdown_index]
    result['volatility_stress_x_ROC20'] = volatility_stress * pd.to_numeric(
        result['ROC20'], errors='coerce'
    )
    result['drawdown_stress_x_STD20'] = drawdown_stress * pd.to_numeric(
        result['STD20'], errors='coerce'
    )
    return result


def add_causal_market_features(df):
    """增加四个只依赖当日及历史数据的市场状态特征。"""
    required = {'日期', '涨跌幅'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"市场状态特征缺少字段: {sorted(missing)}")

    result = df.copy()
    result['日期'] = pd.to_datetime(result['日期']).dt.normalize()
    candidate = pd.Series(True, index=result.index)
    if 'is_member' in result.columns:
        candidate &= result['is_member'].fillna(False).astype(bool)
    if 'is_tradable' in result.columns:
        candidate &= result['is_tradable'].fillna(False).astype(bool)

    market_rows = result.loc[candidate, ['日期', '涨跌幅']].copy()
    market_rows['_return'] = (
        pd.to_numeric(market_rows['涨跌幅'], errors='coerce') / 100.0
    )
    daily = (
        market_rows.dropna(subset=['_return'])
        .groupby('日期', as_index=False)
        .agg(
            market_equal_weight_return=('_return', 'mean'),
            breadth_positive=('_return', lambda values: float((values > 0).mean())),
        )
        .sort_values('日期')
        .reset_index(drop=True)
    )
    if daily.empty:
        raise ValueError('动态候选池中没有可用于计算市场状态的有效收益')

    daily_return = daily['market_equal_weight_return']
    wealth = (1.0 + daily_return).cumprod()
    daily['market_return_20'] = (
        (1.0 + daily_return).rolling(20, min_periods=20).apply(np.prod, raw=True)
        - 1.0
    )
    daily['market_volatility_20'] = daily_return.rolling(
        20, min_periods=20
    ).std(ddof=0)
    daily['market_drawdown_60'] = (
        wealth / wealth.rolling(60, min_periods=60).max() - 1.0
    )
    daily['market_breadth_20'] = daily['breadth_positive'].rolling(
        20, min_periods=20
    ).mean()

    return result.merge(
        daily[['日期', *MARKET4_FEATURES]],
        on='日期',
        how='left',
        validate='many_to_one',
    )


def returns_to_relevance(day_targets):
    """将收益转换为平均秩；收益越高，relevance越高，并列值同秩。"""
    values = np.asarray(day_targets)
    if values.ndim != 1:
        raise ValueError("day_targets必须是一维数组")
    if not np.isfinite(values).all():
        raise ValueError("day_targets包含非有限值")
    return rankdata(values, method="average").astype(np.float32)


# 特征工程
def engineer_features_baseline24(df):
    """只计算baseline24实际使用的24项特征。"""
    try:
        import talib
    except ImportError:
        print("请安装TA-Lib库: pip install TA-Lib")
        raise

    result = df.copy()
    high = result['最高'].astype(float)
    low = result['最低'].astype(float)
    close = result['收盘'].astype(float)
    volume = result['成交量'].astype(float)

    # 直接计算固定24项特征，不构造任何未使用的候选特征。
    for window in [5, 10, 20, 60]:
        result[f'STD{window}'] = (
            talib.STDDEV(close, timeperiod=window) / (close + 1e-12)
        )
    for window in [5, 20, 60]:
        result[f'ROC{window}'] = close.shift(window) / (close + 1e-12)
        result[f'RANK{window}'] = close.rolling(window).rank(pct=True)

    result['MA20'] = talib.SMA(close, timeperiod=20) / (close + 1e-12)
    min_low_20 = low.rolling(20).min()
    max_high_20 = high.rolling(20).max()
    result['RSV20'] = (
        (close - min_low_20) / (max_high_20 - min_low_20 + 1e-12)
    )

    for window in [5, 20]:
        result[f'VMA{window}'] = (
            talib.SMA(volume, timeperiod=window) / (volume + 1e-12)
        )
    result['VSTD20'] = (
        talib.STDDEV(volume, timeperiod=20) / (volume + 1e-12)
    )
    vol_weighted_ret = (close / close.shift(1) - 1).abs() * volume
    result['WVMA20'] = (
        vol_weighted_ret.rolling(20).std()
        / (vol_weighted_ret.rolling(20).mean() + 1e-12)
    )

    volume_ma_5 = talib.SMA(volume, timeperiod=5)
    volume_ma_20 = talib.SMA(volume, timeperiod=20)
    result['volume_ratio'] = volume_ma_5 / volume_ma_20
    result['volume_change'] = volume.pct_change()
    return_1 = close.pct_change(1)
    result['volatility_10'] = return_1.rolling(10).std()
    result['volatility_20'] = return_1.rolling(20).std()
    result['atr_14'] = talib.ATR(high, low, close, timeperiod=14)

    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    result.fillna(0, inplace=True)
    return result


def process_single_stock(stock_row, data, features, sequence_length, date):
    """处理单只股票的数据，返回序列、目标值和股票索引"""
    stock_code = stock_row['instrument']
    # stock_idx = stock_row['stock_idx']
    
    # 获取该股票历史sequence_length天的数据（包括当天）
    stock_history = data[
        (data['instrument'] == stock_code) & 
        (data['datetime'] <= date)
    ].sort_values('datetime').tail(sequence_length)

    if len(stock_history) == sequence_length:
        seq = stock_history[features].values
        target = stock_row['label']  # 下一天的涨跌幅
        return seq, target, stock_code
    else:
        return None, None, None

def process_single_date(date, data, features, sequence_length):
    """处理单个日期的所有股票数据"""
    try:
        # 获取当天有target的股票（即有下一天数据的股票）
        day_data = data[data['datetime'] == date]
        day_data = day_data.dropna(subset=['label'])  # 确保有target
        
        if len(day_data) < 10:  # 确保至少有10只股票
            return None
            
        # 获取当天所有股票的特征序列
        day_sequences = []
        day_targets = []
        day_stock_indices = []
        
        # 对于单个日期内的股票处理，仍使用串行方式避免过度并行化
        # 因为多进程的开销可能超过收益
        for _, stock_row in day_data.iterrows():
            seq, target, stock_idx = process_single_stock(
                stock_row, data, features, sequence_length, date
            )
            if seq is not None:
                day_sequences.append(seq)
                day_targets.append(target)
                day_stock_indices.append(stock_idx)
        
        if len(day_sequences) >= 10:  # 确保有足够的股票
            # 创建排序标签：涨跌幅越高，相关性得分越高；并列收益同秩。
            day_targets = np.array(day_targets)
            relevance = returns_to_relevance(day_targets)
            
            return {
                'sequences': np.array(day_sequences),
                'targets': day_targets,
                'relevance': relevance,
                'stock_indices': day_stock_indices,
                'date': date
            }
        else:
            return None
            
    except Exception as e:
        print(f"处理日期 {date} 时出错: {e}")
        return None

def create_ranking_dataset_multiprocess(data, features, sequence_length, ranking_data_path=None, max_workers=None):
    """
    输入：股票历史数据 DataFrame，特征列名列表，序列长度，排名数据保存路径，最大工作进程数
    输出：排序数据集，格式为：(sequences, targets, relevance_scores, stock_indices)
    - sequences: List of np.array, 每个元素形状为 (num_stocks, sequence_length, num_features)
    - targets: List of np.array, 每个元素形状为 (num_stocks,)
    - relevance_scores: List of np.array, 每个元素形状为 (num_stocks,)
    - stock_indices: List of List, 每个元素为对应股票的索引列表
    """
    """多进程版本的排序数据集创建函数"""
    if ranking_data_path is not None:
        # 如果指定了ranking_data_path，尝试加载已有的数据集
        if os.path.exists(ranking_data_path):
            print(f"加载已有的排序数据集: {ranking_data_path}")
            return joblib.load(ranking_data_path)
    """
    创建排序数据集，按日期组织数据，每个样本包含同一天所有股票的特征和涨跌幅排序
    使用多线程加速处理
    """
    sequences = []
    targets = []
    relevance_scores = []
    stock_indices = []
    
    print("正在创建排序数据集（多线程版本）...")
    
    # 获取所有日期，确保有足够的历史数据
    all_dates = sorted(data['datetime'].unique())
    min_date_for_sequences = all_dates[sequence_length-1]  # 确保有足够历史数据
    
    # 只使用有足够历史数据的日期
    valid_dates = [date for date in all_dates if date >= min_date_for_sequences]
    
    print(f"总日期数: {len(all_dates)}, 有效日期数: {len(valid_dates)}")
    
    # 设置最大工作进程数
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    from tqdm import tqdm
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 10)
    
    print(f"使用 {max_workers} 个进程处理数据")
    
    # 分批处理日期以避免内存问题
    processed_count = 0
        
    # 使用进程池并行处理日期批次
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 创建处理函数的偏函数
            process_func = partial(process_single_date,
                                    data=data,
                                    features=features,
                                    sequence_length=sequence_length)
            
            # 并行处理批次中的所有日期
            futures = [executor.submit(process_func, date) for date in valid_dates]
            
            # 收集结果
            for future in tqdm(futures, desc="Processing dates", total=len(valid_dates)):
                try:
                    result = future.result(timeout=60)  # 设置超时
                    if result is not None:
                        sequences.append(result['sequences'])
                        targets.append(result['targets'])
                        relevance_scores.append(result['relevance'])
                        stock_indices.append(result['stock_indices'])
                        processed_count += 1
                except Exception as e:
                    print(f"处理某个日期时出错: {e}")
                    continue
                    
    except Exception as e:
        print(f"进程池处理出错，回退到串行处理: {e}")
        # 如果多进程出错，回退到串行处理
        for date in tqdm(valid_dates, desc="串行处理"):
            result = process_single_date(date, data, features, sequence_length)
            if result is not None:
                sequences.append(result['sequences'])
                targets.append(result['targets'])
                relevance_scores.append(result['relevance'])
                stock_indices.append(result['stock_indices'])
                processed_count += 1
    
    print(f"成功创建 {len(sequences)} 个训练样本")
    if len(sequences) > 0:
        print(f"每个样本平均包含 {np.mean([len(seq) for seq in sequences]):.1f} 只股票")
    
    # 将四个数据保存下来，下次直接读取
    if ranking_data_path:
        joblib.dump((sequences, targets, relevance_scores, stock_indices), ranking_data_path)
        print(f"数据集已保存到: {ranking_data_path}")
    
    return sequences, targets, relevance_scores, stock_indices

def create_dataset(data, features, sequence_length, ranking_data_path=None):
    """保持原有接口，但内部调用新的排序数据集创建函数"""
    return create_ranking_dataset_multiprocess(data, features, sequence_length, ranking_data_path)

def create_ranking_dataset_vectorized(
    data,
    features,
    sequence_length,
    ranking_data_path=None,
    min_window_end_date=None,
    max_window_end_date=None,
    return_dates=False,
    min_stocks=10,
):
    """
    向量化加速版本：预计算每只股票的所有滑动窗口，再按日期聚合。
    保持与原函数完全相同的输出格式。
    """
    # if ranking_data_path and os.path.exists(ranking_data_path):
    #     print(f"加载已有的排序数据集: {ranking_data_path}")
    #     return joblib.load(ranking_data_path)

    print("正在创建排序数据集（向量化加速版本）...")
    # data.rename(columns={'stock_idx': 'instrument'}, inplace=True)
    data = data.copy()
    data.rename(columns={'日期': 'datetime'}, inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'])

    # 1. 确保数据按股票和时间排序
    data = data.sort_values(['instrument', 'datetime']).reset_index(drop=True)
    
    # 2. 保留完整历史作为序列上下文。只有窗口结束日需要有效 label；
    # PIT 数据中的非成分时期仍是公开历史，调入后构造序列时需要这些行。
    if 'label' not in data.columns:
        raise ValueError("排序数据缺少 label 列")
    if 'is_member' not in data.columns:
        data['is_member'] = True
    if 'is_tradable' not in data.columns:
        data['is_tradable'] = True
    
    # 3. 为每只股票生成所有滑动窗口
    # label 已在上游通过 shift(-1/-5) 构造并删除缺失值。
    # 因此这里只需检查历史窗口长度，不能再次删除末尾5个有效标签日。
    all_windows = []  # 每个元素: (end_date, stock_code, sequence, target)

    print("Step 1: 为每只股票生成滑动窗口...")
    grouped = data.groupby('instrument')
    
    for stock_code, group in tqdm(grouped, desc="Processing stocks"):
        if len(group) < sequence_length:
            continue
        
        # 提取特征和 label
        feature_values = group[features].values.astype(np.float32)  # (T, F)
        labels = group['label'].values.astype(np.float32)           # (T,)
        dates = group['datetime'].values                            # (T,)
        candidates = (
            group['is_member'].fillna(False).astype(bool).values
            & group['is_tradable'].fillna(False).astype(bool).values
        )

        # 生成滑动窗口：从第 sequence_length-1 行开始（0-indexed）
        num_windows = len(group) - sequence_length + 1
        for i in range(num_windows):
            end_idx = i + sequence_length - 1
            if not candidates[end_idx] or not np.isfinite(labels[end_idx]):
                continue

            seq = feature_values[i : i + sequence_length]   # (L, F)
            target = labels[end_idx]                        # label 对应窗口最后一天的未来5日收益
            end_date = dates[end_idx]                       # 窗口结束日期（即预测日）
            all_windows.append((end_date, stock_code, seq, target))

    # 4. 转为 DataFrame 便于按日期聚合
    print("Step 2: 按日期聚合窗口...")
    window_df = pd.DataFrame(all_windows, columns=['date', 'stock_code', 'seq', 'target'])

    # 5. 按 date 分组，构建每日样本
    sequences = []
    targets = []
    relevance_scores = []
    stock_indices = []

    print("Step 3: 构建每日样本并计算 relevance...")
    grouped_by_date = window_df.groupby('date')

    if min_window_end_date is not None:
        min_window_end_date = pd.to_datetime(min_window_end_date)
    if max_window_end_date is not None:
        max_window_end_date = pd.to_datetime(max_window_end_date)
    
    for date, group in tqdm(grouped_by_date, desc="Aggregating by date"):
        if min_window_end_date is not None and pd.to_datetime(date) < min_window_end_date:
            continue
        if max_window_end_date is not None and pd.to_datetime(date) > max_window_end_date:
            continue

        if len(group) < min_stocks:
            continue
        
        # 提取数据
        day_seqs = np.stack(group['seq'].values)          # (N, L, F)
        day_targets = group['target'].values              # (N,)
        day_stocks = group['stock_code'].tolist()         # [str]

        relevance = returns_to_relevance(day_targets)

        sequences.append(day_seqs)
        targets.append(day_targets)
        relevance_scores.append(relevance)
        stock_indices.append(day_stocks)

    print(f"成功创建 {len(sequences)} 个训练样本")
    if len(sequences) > 0:
        avg_stocks = np.mean([len(seq) for seq in sequences])
        print(f"每个样本平均包含 {avg_stocks:.1f} 只股票")

    # 6. 保存
    # if ranking_data_path:
    #     joblib.dump((sequences, targets, relevance_scores, stock_indices), ranking_data_path)
    #     print(f"数据集已保存到: {ranking_data_path}")

    if return_dates:
        sample_dates = [
            pd.Timestamp(date)
            for date, group in grouped_by_date
            if (min_window_end_date is None or pd.Timestamp(date) >= min_window_end_date)
            and (max_window_end_date is None or pd.Timestamp(date) <= max_window_end_date)
            and len(group) >= min_stocks
        ]
        return sequences, targets, relevance_scores, stock_indices, sample_dates

    return sequences, targets, relevance_scores, stock_indices
