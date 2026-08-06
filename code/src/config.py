# 配置参数
sequence_length = 60
feature_num = 'baseline24_market4'
config = {
    'seed': 42,             # 默认随机种子；多种子实验会显式覆盖
    'sequence_length': sequence_length,   # 使用过去60个交易日的数据（排序任务可以用稍短的序列）
    'd_model': 256,          # d_model=128消融失败，恢复主线容量
    'nhead': 4,             # 注意力头数量
    'num_layers': 3,        # Transformer层数
    'dim_feedforward': 512, # 前馈网络维度
    'batch_size': 4,        # 排序任务batch_size可以小一些，因为每个batch包含更多股票
    'num_epochs': 50,       # 排序任务可能需要更多epochs
    # 学习率计划总长度独立于实际训练轮数。固定epoch外层重训仍复现
    # 50轮内层训练轨迹的前k轮。
    'scheduler_total_epochs': 50,
    'early_stopping_patience': 10,   # 正式训练连续10轮无提升时早停
    'learning_rate': 1e-5,
    'dropout': 0.1,
    'model_variant': 'hybrid',
    'feature_num': feature_num,
    'exclude_features': ['instrument'], # 股票编号仅用于分组，不进入模型
    'include_features': [         # baseline24 + 市场4，共28维
        '振幅', 'STD5', 'STD10', 'STD20', 'STD60',
        'volatility_10', 'volatility_20', 'atr_14',
        '成交额', '换手率', 'volume_ratio', 'volume_change',
        'VMA5', 'VMA20', 'VSTD20', 'WVMA20',
        'ROC5', 'ROC20', 'ROC60', 'RANK5', 'RANK20', 'RANK60',
        'RSV20', 'MA20',
        'market_return_20', 'market_volatility_20',
        'market_drawdown_60', 'market_breadth_20',
    ],
    'feature_transform': 'cross_sectional_rank',
    'max_grad_norm': 5.0,
    'listwise_target_type': 'normalized_rank_softmax',
    'listwise_temperature': 1.0,

    # Top-5组合主目标：头部与非头部的可分性是主损失，全横截面排序只作正则。
    'head_k': 5,
    'head_loss_weight': 1.0,
    'global_rank_weight': 0.2,
    'stability_weight': 0.05,
    # 防抄近路正则：惩罚模型分数与日内横截面波动率指数过度相关。
    # 默认关闭以保证历史实验可复现；新实验通过CLI显式设为0.05。
    'shortcut_correlation_weight': 0.0,
    'shortcut_volatility_features': [
        '振幅', 'STD5', 'STD10', 'STD20', 'STD60',
        'volatility_10', 'volatility_20', 'atr_14', '换手率',
    ],
    'pairwise_weight': 1, # 仅供历史全排序基线复现
    'base_weight': 1.0, # 非top-k样本权重
    'top5_weight': 2.0, # top-5样本权重（应大于base_weight）
    # head_focused为新主线；weighted_ranknet(_stability)仅供历史基线复现。
    'ranking_loss_type': 'head_focused',
    # 比赛阶段严格复现score_self.py：未来第1条记录开盘 -> 第5条记录开盘。
    'label_mode': 'contest_score',
    'model_selection_metric': 'top5_return',

    'output_dir': f'../../model/{sequence_length}_{feature_num}',
    'data_path': '../../data',
    # 新实验唯一允许的数据源；历史固定股票池文件仅用于复现实验。
    'data_file': 'data/hs300_pit/model_data.csv',
    # 仅用最新信号日向前3个自然年的数据做70%/15%/15%时间切分。
    # 更早的数据仍可作为60日序列的历史上下文，但不进入train/validation/test目标日期。
    'data_lookback_years': 3,
    'split_ratios': [0.70, 0.15, 0.15],
    'gap_days': 5,
    # 比赛阶段默认：无验证集，一个信号日，随后5个交易日按score_self.py评分。
    # 论文滚动/嵌套验证需在对应实验入口显式切换协议。
    'training_protocol': 'competition_5day',
    # 默认保持自然采样；分层实验必须显式启用，首轮弱市占比冻结为30%。
    'sampling_strategy': 'natural',
    'weak_target_share': 0.30,

    # ============ TCN 参数 ============
    'tcn_channels': 64,
    'tcn_kernel_size': 3,        # 卷积核大小
    'tcn_num_layers': 5,         # dilation=1,2,4,8,16，实际感受野63日
    'tcn_min_receptive_field': sequence_length,  # 强制覆盖完整60日输入
    'tcn_dropout': 0.1,          # TCN dropout
    # 新主线不跨股票聚合统计量；旧checkpoint缺少该字段时仍按batchnorm加载。
    'tcn_norm_type': 'layernorm',

    # ============ iTransformer 参数 ============
    'it_nhead': 4,               # iTransformer注意力头数
    'it_num_layers': 2,          # iTransformer编码器层数
    'it_dim_feedforward': 512,   # iTransformer FFN维度
    'it_variable_identity_encoding': 'learned',  # 真实变量token的可学习身份编码

    # ============ 融合参数 ============
    'fusion_dropout': 0.1,       # 融合层 dropout

    # ============ 论文INFO超参优化器（默认关闭） ============
    'use_info': False,
    'info_search_budget': 5,
    'info_num_trials': 3,
    'info_stability_lambda': 0.5,
    'info_max_epochs': 5,
    'info_require_positive': True,

    # ============ 论文辅助损失（当前消融选择权重0） ============
    'auxiliary_loss_weight': 0.0,

    # ============ 市场状态软门控（开发方案） ============
    # 用市场4特征决定原模型分数与冻结防御因子分数的连续混合比例。
    'soft_gate_enabled': True,
    'soft_gate_hidden_dim': 16,
    # 以可解释的因果压力先验启动门控，再由排序损失学习残差。旧版从常数
    # sigmoid(-2)=0.119 启动且与主模型共用 1e-5 学习率，50 轮内几乎不动。
    'soft_gate_rule_mode': True,
    'soft_gate_initial_bias': -1.5,
    'soft_gate_stress_strength': 1.25,
    'soft_gate_latest_weight': 0.75,
    'soft_gate_history_weight': 0.25,
    'soft_gate_threshold_low_quantile': 0.70,
    'soft_gate_threshold_high_quantile': 0.90,
    'soft_gate_trigger_mode': 'two_of_four',
    'soft_gate_min_abnormal_components': 2,
    'soft_gate_rule_min': 0.30,
    'soft_gate_rule_max': 0.50,
    'soft_gate_cap': 0.60,
    'soft_gate_residual_fraction': 0.20,
    'soft_gate_learning_rate_multiplier': 5.0,
    'soft_gate_market_features': [
        'market_return_20', 'market_volatility_20',
        'market_drawdown_60', 'market_breadth_20',
    ],
    'soft_gate_low_volatility_features': [
        '振幅', 'STD5', 'STD10', 'STD20', 'STD60',
        'volatility_10', 'volatility_20', 'atr_14',
    ],
    'soft_gate_reversal_feature': 'ROC60',
    'soft_gate_low_volatility_weight': 0.5,
    'soft_gate_reversal_weight': 0.5,

}
