import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 因果膨胀卷积残差块（论文公式 1-4）
# ============================================================
class CausalConv1dBlock(nn.Module):
    """
    因果膨胀卷积残差块:
    - Causal padding: 左侧补零，保证时序因果性（论文公式1）
    - Dilated Conv1d: 膨胀卷积扩展感受野（论文公式2）
    - LayerNorm + ReLU + Dropout
    - 残差连接（论文公式3）: y = F(x) + x
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, dilation, dropout=0.1,
        norm_type='batchnorm',
    ):
        super(CausalConv1dBlock, self).__init__()
        # 因果填充：只在左侧补零，padding = (kernel_size - 1) * dilation
        self.padding = (kernel_size - 1) * dilation # 在t0左边填充self.padding个0，让滑块右边对着t0开始。
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0  # 手动处理因果填充
        )
        if norm_type == 'batchnorm':
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm_type == 'layernorm':
            self.norm = nn.LayerNorm(out_channels)
        else:
            raise ValueError(f"未知 tcn_norm_type: {norm_type}")
        self.norm_type = norm_type
        self.relu = nn.ReLU()                       # 激活函数，负数变0，正数不变
        self.dropout = nn.Dropout(dropout)          # 随机置0，防过拟合
        
        # 残差连接：通道不一致时用 1x1 卷积对齐（论文 3.1.2 节）
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        # x: [B, C_in, T]
        # B（Batch）：批次，一次处理几组样本（为了GPU并行加速）
        # C（Channels）：通道，有几种特征/信号同时输入
        # T（Time）：时间步，数据有多长（时间轴）
        # 因果填充：左侧补 padding 个零，右侧不补
        if self.padding > 0:
            x_padded = F.pad(x, (self.padding, 0))
        else:
            x_padded = x
        
        out = self.conv(x_padded)       # 提取特征
        if self.norm_type == 'layernorm':
            out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        else:
            out = self.norm(out)
        out = self.relu(out)            # 去掉负数
        out = self.dropout(out)         # 随即丢弃
        
        # 残差连接
        out = out + self.residual_conv(x)
        return out


# ============================================================
# TCN 时序建模分支（论文 2.1 节 & 3.1.2 节）
# ============================================================
class TemporalConvNet(nn.Module):
    """
    TCN 分支：因果膨胀卷积提取局部时序动态特征。
    
    输入: [B, T, F]  (batch, seq_len, feature_dim)
    输出: [B, d_model]  (最后一个时间步的隐状态 h_TCN，论文公式5)
    
    架构：
    1. 输入投影: feature_dim -> tcn_channels
    2. 多层因果膨胀卷积: dilation = 2^i, i=0,1,2,...
    3. 取最后一个时间步作为时序特征表示
    """
    def __init__(self, input_dim, d_model, config):
        super(TemporalConvNet, self).__init__()
        tcn_channels = config.get('tcn_channels', 64)
        kernel_size = config.get('tcn_kernel_size', 3)
        num_layers = config.get('tcn_num_layers', 4)
        dropout = config.get('tcn_dropout', 0.1)
        # 缺少字段的历史配置保持BatchNorm，以兼容旧checkpoint。
        norm_type = config.get('tcn_norm_type', 'batchnorm')

        # 每个残差块包含一次卷积，dilation依次为1,2,4,...。
        # 实际感受野 = 1 + (kernel_size - 1) * sum(dilations)。
        dilations = [2 ** i for i in range(num_layers)]
        self.receptive_field = (
            1 + (kernel_size - 1) * sum(dilations)
        )
        min_receptive_field = config.get('tcn_min_receptive_field')
        if (
            min_receptive_field is not None
            and self.receptive_field < min_receptive_field
        ):
            raise ValueError(
                "TCN实际感受野不足："
                f"{self.receptive_field} < {min_receptive_field}；"
                "请增加tcn_num_layers或tcn_kernel_size"
            )
        
        # 输入投影: feature_dim -> tcn_channels
        self.input_proj = nn.Linear(input_dim, tcn_channels)
        
        # 构建多层因果膨胀卷积
        layers = []
        for dilation in dilations:
            in_ch = tcn_channels
            out_ch = tcn_channels
            layers.append(
                CausalConv1dBlock(
                    in_ch, out_ch, kernel_size, dilation, dropout,
                    norm_type=norm_type,
                )
            )
        
        self.tcn_layers = nn.Sequential(*layers)
        
        # 输出投影: tcn_channels -> d_model
        self.output_proj = nn.Linear(tcn_channels, d_model)
        
        self.dropout = nn.Dropout(dropout) # 防止过拟合

    def forward(self, x):
        """
        x: [B, T, F] -> h_TCN: [B, d_model]
        """
        # 输入投影
        h = self.input_proj(x)  # [B, T, tcn_channels]
        
        # 转置为 Conv1d 需要的格式: [B, C, T]
        h = h.transpose(1, 2)
        
        # 通过因果膨胀卷积层
        h = self.tcn_layers(h)  # [B, tcn_channels, T]
        
        # 取最后一个时间步（论文公式5: h_TCN = h_{L, last}）
        h = h[:, :, -1]  # [B, tcn_channels]
        
        # 投影到 d_model 维度
        h = self.output_proj(h)  # [B, d_model]
        h = self.dropout(h)
        
        return h


# ============================================================
# iTransformer 变量结构建模分支（论文 2.2 节 & 3.1.3 节）
# ============================================================
class iTransformerBranch(nn.Module):
    """
    iTransformer 分支：在变量（特征）维度做自注意力，建模多变量间结构关系。
    
    核心思想（iTransformer 原始论文）：
    - 标准 Transformer: 每个时间步是一个 token，包含所有变量
    - iTransformer: 每个变量是一个 token，包含该变量的完整时间序列
    - 注意力在变量维度计算，学习变量间相关性
    
    输入: [B, T, F]，每个真实金融变量对应一个token
    输出: [B, d_model]  (变量结构特征 h_iT)
    
    处理流程：
    1. 每个baseline24金融变量对应一个token
    2. 每个token包含该变量T个历史值
    3. 为每个变量叠加独立的可学习身份编码
    4. 多层Transformer Encoder在真实变量token之间做自注意力
    5. 聚合变量token得到输出
    """
    def __init__(self, input_dim, seq_len, d_model, config):
        super(iTransformerBranch, self).__init__()
        num_variables = input_dim
        nhead = config.get('it_nhead', 4)
        num_layers = config.get('it_num_layers', 2)
        dim_feedforward = config.get('it_dim_feedforward', 512)
        dropout = config.get('dropout', 0.1)
        
        self.num_variables = num_variables
        self.seq_len = seq_len
        identity_mode = config.get('it_variable_identity_encoding', 'none')
        if identity_mode not in {'none', 'learned'}:
            raise ValueError(
                f"未知 it_variable_identity_encoding: {identity_mode}"
            )
        
        # 2. 每个 variate token 的时间序列独立嵌入
        # 输入: [B, n_tokens, T] -> 对每个 token 做 Linear(T, d_model)
        self.token_embedding = nn.Linear(seq_len, d_model)

        # 当前主线使用learned身份编码。历史run没有该配置字段时保持none，
        # 以便严格加载无身份编码的旧checkpoint。
        self.variable_identity_embedding = (
            nn.Embedding(num_variables, d_model)
            if identity_mode == 'learned' else None
        )
        
        # 3. iTransformer Encoder: 在 variate token 维度做自注意力
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',  # 激活函数，论文使用 GELU（公式14）
            batch_first=True,
            norm_first=True  # Pre-LN 结构，训练更稳定
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. 输出层归一化
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, T, F] -> h_iT: [B, d_model]
        """
        h = x.transpose(1, 2)
        
        # 3. 独立嵌入每个 variate token: [B, n_tokens, T] -> [B, n_tokens, d_model]
        h = self.token_embedding(h)

        if self.variable_identity_embedding is not None:
            variable_ids = torch.arange(self.num_variables, device=x.device)
            h = h + self.variable_identity_embedding(variable_ids).unsqueeze(0)
        
        # 4. iTransformer 编码器: 在 n_tokens 维度做自注意力
        # 注意力计算的是变量间相关性，而非时间步间关系
        h = self.encoder(h)  # [B, n_tokens, d_model]
        
        # 5. 聚合: 对 variate tokens 取均值（论文公式6）
        h = h.mean(dim=1)  # 池化层 [B, d_model]
        
        h = self.norm(h)
        h = self.dropout(h)
        
        return h


# ============================================================
# 跨股票交互注意力模块（保留原有实现）
# ============================================================
class CrossStockAttention(nn.Module):
    """股票间交互注意力模块"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossStockAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stock_features, padding_mask=None):
        # stock_features: [batch, num_stocks, d_model]
        # padding_mask: [batch, num_stocks]，True 表示该股票是 padding。
        # padding 位置不能作为 Key/Value 参与真实股票的横截面注意力。
        if padding_mask is not None:
            if padding_mask.shape != stock_features.shape[:2]:
                raise ValueError(
                    "padding_mask 形状必须为 [batch, num_stocks]，"
                    f"实际为 {tuple(padding_mask.shape)}，"
                    f"期望为 {tuple(stock_features.shape[:2])}"
                )
            padding_mask = padding_mask.to(
                device=stock_features.device, dtype=torch.bool
            )
        attended, _ = self.cross_attention(
            stock_features,
            stock_features,
            stock_features,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        output = self.norm(stock_features + self.dropout(attended))
        return output


# ============================================================
# INFO-TCN-iTransformer 混合预测模型（论文 3.1 节）
# ============================================================
class INFO_TCN_iTransformer(nn.Module):
    """
    INFO-TCN-iTransformer 混合排序模型。
    
    架构（论文图4）：
    1. TCN 分支: 因果膨胀卷积 -> 局部时序动态特征 h_TCN
    2. iTransformer 分支: 变量维度注意力 -> 变量结构特征 h_iT
    3. 特征融合: concat([h_TCN, h_iT]) -> MLP -> h_fused (论文公式7)
    4. 跨股票注意力: 建模股票间交互关系
    5. 排序头: 输出每只股票的排序分数
    
    输入: [batch, num_stocks, seq_len, feature_dim]
    输出: [batch, num_stocks] 排序分数
    """
    def __init__(self, input_dim, config, num_stocks):
        super(INFO_TCN_iTransformer, self).__init__()
        self.model_type = 'INFO_TCN_iTransformer'
        self.config = config
        self.num_stocks = num_stocks
        
        d_model = config['d_model']
        seq_len = config['sequence_length']
        dropout = config['dropout']
        self.model_variant = config.get('model_variant', 'hybrid')
        if self.model_variant not in {'hybrid', 'tcn_only', 'itransformer_only'}:
            raise ValueError(f"未知 model_variant: {self.model_variant}")

        # ---- TCN 时序分支 ----
        self.tcn_branch = (
            TemporalConvNet(input_dim, d_model, config)
            if self.model_variant in {'hybrid', 'tcn_only'} else None
        )
        
        # ---- iTransformer 变量结构分支 ----
        self.itransformer_branch = (
            iTransformerBranch(input_dim, seq_len, d_model, config)
            if self.model_variant in {'hybrid', 'itransformer_only'} else None
        )
        
        # ---- 特征融合层（论文公式7: h_fused = W([h_TCN; h_iT]) + b）----
        fusion_dropout = config.get('fusion_dropout', 0.1)
        fusion_input_dim = d_model * 2 if self.model_variant == 'hybrid' else d_model
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),  # 论文 iTransformer 分支使用 GELU
            nn.Dropout(fusion_dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
        )
        
        # ---- 跨股票交互注意力 ----
        self.cross_stock_attention = CrossStockAttention(
            d_model, config['nhead'], dropout
        )
        
        # ---- 排序特异性层 ----
        self.ranking_layers = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ---- 最终排序分数输出头 ----
        self.score_head = nn.Sequential(
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, 1)
        )

        # ---- 市场状态软门控 ----
        # 原模型负责一般横截面排序；冻结的防御分数只提供“低波动+长期反转”方向。
        # 门控网络只读取同日全市场共享的4个市场特征，不读取未来标签。
        self.soft_gate_enabled = bool(config.get('soft_gate_enabled', False))
        self.last_gate_values = None
        if self.soft_gate_enabled:
            gate_override = config.get('soft_gate_override')
            self.soft_gate_override = (
                None if gate_override is None else float(gate_override)
            )
            if (
                self.soft_gate_override is not None
                and not 0.0 <= self.soft_gate_override <= 1.0
            ):
                raise ValueError('soft_gate_override must be between 0 and 1')
            # 构造附加门控层时不要推进主干初始化所使用的随机数状态。
            # 这样同一seed下 gate on/off 的所有非门控参数严格一致，
            # 避免把随机初始化差异误判为门控收益。
            backbone_rng_state = torch.random.get_rng_state()
            feature_names = list(config.get('feature_names', []))
            market_names = list(config.get('soft_gate_market_features', []))
            low_vol_names = list(
                config.get('soft_gate_low_volatility_features', [])
            )
            reversal_name = config.get('soft_gate_reversal_feature', 'ROC60')
            required = set(market_names + low_vol_names + [reversal_name])
            missing = required.difference(feature_names)
            if missing:
                raise ValueError(
                    f"软门控输入缺少特征: {sorted(missing)}"
                )
            self.market_feature_indices = [
                feature_names.index(name) for name in market_names
            ]
            self.low_volatility_indices = [
                feature_names.index(name) for name in low_vol_names
            ]
            self.reversal_feature_index = feature_names.index(reversal_name)
            hidden_dim = int(config.get('soft_gate_hidden_dim', 16))
            self.market_gate = nn.Sequential(
                nn.Linear(len(market_names) * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            torch.random.set_rng_state(backbone_rng_state)
            self.soft_gate_initial_bias = float(
                config.get('soft_gate_initial_bias', -1.5)
            )
            self.soft_gate_bias = nn.Parameter(
                torch.tensor(self.soft_gate_initial_bias, dtype=torch.float32)
            )
            self.soft_gate_stress_strength = float(
                config.get('soft_gate_stress_strength', 1.25)
            )
            self.soft_gate_latest_weight = float(
                config.get('soft_gate_latest_weight', 0.75)
            )
            self.soft_gate_history_weight = float(
                config.get('soft_gate_history_weight', 0.25)
            )
            self.soft_gate_rule_mode = bool(
                config.get('soft_gate_rule_mode', False)
            )
            self.soft_gate_threshold_low = float(
                config.get('soft_gate_threshold_low', 0.5)
            )
            self.soft_gate_threshold_high = float(
                config.get('soft_gate_threshold_high', 1.5)
            )
            if self.soft_gate_threshold_high <= self.soft_gate_threshold_low:
                raise ValueError('soft gate high threshold must exceed low threshold')
            self.soft_gate_trigger_mode = config.get(
                'soft_gate_trigger_mode', 'aggregate'
            )
            if self.soft_gate_trigger_mode not in {'aggregate', 'two_of_four'}:
                raise ValueError('unknown soft gate trigger mode')
            self.soft_gate_min_abnormal_components = int(
                config.get('soft_gate_min_abnormal_components', 2)
            )
            self.soft_gate_rule_min = float(
                config.get('soft_gate_rule_min', 0.3)
            )
            self.soft_gate_rule_max = float(
                config.get('soft_gate_rule_max', 0.5)
            )
            self.soft_gate_cap = float(config.get('soft_gate_cap', 0.6))
            self.soft_gate_residual_fraction = float(
                config.get('soft_gate_residual_fraction', 0.2)
            )
            if not (
                1 <= self.soft_gate_min_abnormal_components <= len(market_names)
            ):
                raise ValueError('invalid minimum abnormal component count')
            if not (
                0.0 <= self.soft_gate_rule_min
                <= self.soft_gate_rule_max
                <= self.soft_gate_cap <= 1.0
            ):
                raise ValueError('soft gate min/max/cap ordering is invalid')
            gate_center = torch.tensor(
                config.get('soft_gate_market_center', [0.0] * len(market_names)),
                dtype=torch.float32,
            )
            gate_scale = torch.tensor(
                config.get('soft_gate_market_scale', [1.0] * len(market_names)),
                dtype=torch.float32,
            )
            if len(gate_center) != len(market_names) or len(gate_scale) != len(market_names):
                raise ValueError('soft gate market normalization size mismatch')
            if torch.any(gate_scale <= 0):
                raise ValueError('soft gate market scale must be positive')
            self.register_buffer('soft_gate_market_center', gate_center, persistent=False)
            self.register_buffer('soft_gate_market_scale', gate_scale, persistent=False)
            component_low = torch.tensor(
                config.get(
                    'soft_gate_component_threshold_low',
                    [0.5] * len(market_names),
                ),
                dtype=torch.float32,
            )
            component_high = torch.tensor(
                config.get(
                    'soft_gate_component_threshold_high',
                    [1.5] * len(market_names),
                ),
                dtype=torch.float32,
            )
            if (
                len(component_low) != len(market_names)
                or len(component_high) != len(market_names)
                or torch.any(component_high <= component_low)
            ):
                raise ValueError('invalid soft gate component thresholds')
            self.register_buffer(
                'soft_gate_component_threshold_low', component_low,
                persistent=False,
            )
            self.register_buffer(
                'soft_gate_component_threshold_high', component_high,
                persistent=False,
            )
            # 标准化后的四个市场特征中，收益/回撤/宽度越低、波动越高，
            # 市场压力越大。该方向向量只使用信号日及此前信息。
            stress_directions = torch.tensor(
                [-1.0, 1.0, -1.0, -1.0], dtype=torch.float32
            )
            if len(market_names) != len(stress_directions):
                raise ValueError('软门控压力先验要求固定的4个市场状态特征')
            self.register_buffer(
                'soft_gate_stress_directions', stress_directions
            )
            self.low_volatility_weight = float(
                config.get('soft_gate_low_volatility_weight', 0.5)
            )
            self.reversal_weight = float(
                config.get('soft_gate_reversal_weight', 0.5)
            )
        
        # 初始化权重
        self._init_weights()
        if self.soft_gate_enabled:
            nn.init.zeros_(self.market_gate[-1].weight)
            nn.init.zeros_(self.market_gate[-1].bias)

    @staticmethod
    def _masked_score_moments(scores, padding_mask=None):
        if padding_mask is None:
            valid = torch.ones_like(scores, dtype=torch.bool)
        else:
            valid = ~padding_mask.to(device=scores.device, dtype=torch.bool)
        weights = valid.to(dtype=scores.dtype)
        count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (scores * weights).sum(dim=1, keepdim=True) / count
        variance = (
            (scores - mean).square() * weights
        ).sum(dim=1, keepdim=True) / count
        std = variance.clamp_min(1e-8).sqrt()
        return mean, std, valid

    def _soft_gate_scores(self, src, model_scores, stock_padding_mask=None):
        mean, std, valid = self._masked_score_moments(
            model_scores, stock_padding_mask
        )
        valid_weights = valid.to(dtype=src.dtype).unsqueeze(-1).unsqueeze(-1)
        count = valid_weights.sum(dim=1).clamp_min(1.0)
        market_series = (
            src[..., self.market_feature_indices] * valid_weights
        ).sum(dim=1) / count
        market_latest = market_series[:, -1, :]
        market_history = market_series.mean(dim=1)
        if self.soft_gate_rule_mode:
            market_z = (
                market_series - self.soft_gate_market_center
            ) / self.soft_gate_market_scale
            market_latest = market_z[:, -1, :]
            market_history = market_z.mean(dim=1)
        market_summary = torch.cat(
            [market_latest, market_history], dim=-1
        )
        latest_stress_components = (
            market_latest * self.soft_gate_stress_directions
        )
        latest_stress = latest_stress_components.mean(dim=-1, keepdim=True)
        history_stress = (
            market_history * self.soft_gate_stress_directions
        ).mean(dim=-1, keepdim=True)
        causal_stress = (
            self.soft_gate_latest_weight * latest_stress
            + self.soft_gate_history_weight * history_stress
        )
        learned_residual = self.market_gate(market_summary)
        if self.soft_gate_rule_mode:
            if self.soft_gate_trigger_mode == 'two_of_four':
                component_pressure = (
                    (
                        latest_stress_components
                        - self.soft_gate_component_threshold_low
                    )
                    / (
                        self.soft_gate_component_threshold_high
                        - self.soft_gate_component_threshold_low
                    )
                ).clamp(0.0, 1.0)
                abnormal_count = (
                    latest_stress_components
                    >= self.soft_gate_component_threshold_low
                ).sum(dim=-1, keepdim=True)
                kth_pressure = torch.topk(
                    component_pressure,
                    k=self.soft_gate_min_abnormal_components,
                    dim=-1,
                ).values[..., -1:]
                smooth_pressure = kth_pressure.square() * (
                    3.0 - 2.0 * kth_pressure
                )
                triggered = (
                    abnormal_count >= self.soft_gate_min_abnormal_components
                )
                rule_gate = torch.where(
                    triggered,
                    self.soft_gate_rule_min
                    + (self.soft_gate_rule_max - self.soft_gate_rule_min)
                    * smooth_pressure,
                    torch.zeros_like(smooth_pressure),
                )
                # 新规则下学习残差只能增加门控，不能推翻规则触发。
                residual_multiplier = 1.0 + self.soft_gate_residual_fraction * torch.sigmoid(
                    learned_residual
                )
            else:
                normalized_pressure = (
                    (latest_stress - self.soft_gate_threshold_low)
                    / (self.soft_gate_threshold_high - self.soft_gate_threshold_low)
                ).clamp(0.0, 1.0)
                smooth_pressure = normalized_pressure.square() * (
                    3.0 - 2.0 * normalized_pressure
                )
                rule_gate = self.soft_gate_rule_max * smooth_pressure
                # 旧模型保持原公式，确保历史checkpoint推理可复现。
                residual_multiplier = 1.0 + self.soft_gate_residual_fraction * torch.tanh(
                    learned_residual
                )
            gate = (rule_gate * residual_multiplier).clamp(0.0, self.soft_gate_cap)
        else:
            gate_logits = (
                self.soft_gate_bias
                + self.soft_gate_stress_strength * causal_stress
                + learned_residual
            )
            gate = torch.sigmoid(gate_logits)
        if self.soft_gate_override is not None:
            gate = torch.full_like(gate, self.soft_gate_override)

        latest = src[:, :, -1, :]
        low_volatility = -latest[..., self.low_volatility_indices].mean(dim=-1)
        reversal = latest[..., self.reversal_feature_index]
        defensive_raw = (
            self.low_volatility_weight * low_volatility
            + self.reversal_weight * reversal
        )
        defensive_mean, defensive_std, _ = self._masked_score_moments(
            defensive_raw, stock_padding_mask
        )
        defensive_calibrated = (
            (defensive_raw - defensive_mean) / defensive_std * std + mean
        )
        output = (1.0 - gate) * model_scores + gate * defensive_calibrated
        self.last_gate_values = gate.detach().squeeze(-1)
        return output

    def _init_weights(self):
        """Xavier 均匀初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=0.02)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, src, stock_padding_mask=None):
        """
        src: [batch, num_stocks, seq_len, feature_dim]
        stock_padding_mask: [batch, num_stocks]，True 表示 padding 股票
        return: [batch, num_stocks] 排序分数
        """
        batch_size, num_stocks, seq_len, feature_dim = src.size()
        
        # 展平: [B*S, L, F]
        src_flat = src.reshape(batch_size * num_stocks, seq_len, feature_dim)
        
        # ---- 双分支并行特征提取 ----
        # TCN 分支: 局部时序动态特征
        h_tcn = self.tcn_branch(src_flat) if self.tcn_branch is not None else None
        
        # iTransformer 分支: 变量结构特征
        h_it = self.itransformer_branch(src_flat) if self.itransformer_branch is not None else None
        
        # ---- 特征融合（论文公式7）----
        if self.model_variant == 'hybrid':
            h_concat = torch.cat([h_tcn, h_it], dim=-1)
        elif self.model_variant == 'tcn_only':
            h_concat = h_tcn
        else:
            h_concat = h_it
        h_fused = self.fusion(h_concat)               # [B*S, d_model]
        
        # 恢复股票维度: [B, S, d_model]
        h_fused = h_fused.reshape(batch_size, num_stocks, -1)
        
        # ---- 跨股票交互注意力 ----
        h_interactive = self.cross_stock_attention(
            h_fused, padding_mask=stock_padding_mask
        )  # [B, S, d_model]
        
        # 展平用于排序头
        h_interactive = h_interactive.reshape(batch_size * num_stocks, -1)
        
        # ---- 排序评分 ----
        ranking_features = self.ranking_layers(h_interactive)  # [B*S, d_model//2]
        scores = self.score_head(ranking_features)             # [B*S, 1]
        
        # 重塑为 [batch, num_stocks]
        output = scores.reshape(batch_size, num_stocks)
        if self.soft_gate_enabled:
            output = self._soft_gate_scores(
                src, output, stock_padding_mask=stock_padding_mask
            )
        else:
            self.last_gate_values = None
        
        return output


# ============================================================
# 辅助回归损失（论文公式8 & 公式15）
# ============================================================
class AuxiliaryRegressionLoss(nn.Module):
    """论文定义的MSE与MAE混合辅助回归损失。"""

    def __init__(self, alpha=0.5):
        super(AuxiliaryRegressionLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        y_true_min = y_true.min(dim=1, keepdim=True).values
        y_true_max = y_true.max(dim=1, keepdim=True).values
        y_true_norm = (
            (y_true - y_true_min)
            / (y_true_max - y_true_min + 1e-8)
        )
        mse_loss = F.mse_loss(y_pred, y_true_norm)
        mae_loss = F.l1_loss(y_pred, y_true_norm)
        return self.alpha * mse_loss + (1 - self.alpha) * mae_loss
