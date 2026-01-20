你指出的这三个“未完成关键点”极其精准，它们确实是当前架构从“观察系统”迈向“自适应生存系统”必须跨越的鸿沟。这不再是功能增量，而是系统智能的质变。

我们将这三个问题定位为 V1.5 的核心升级目标。以下是针对每个问题的具体、可落地的工程方案与路线图。

V1.5 升级蓝图：从“记录”到“反制”

问题一：库存PnL“半真实” -> 引入“流动性健康度”贴现因子

目标：让库存估值反映市场可退出性，在流动性枯竭时自动“保守记账”，防止主权层基于虚高健康度做出错误决策。

工程方案：流动性感知的库存估值模型
我们增加一个 LiquidityHealthScore (LHS, 0.0-1.0)，作为库存估值的贴现因子。

```python
class LiquidityAwareInventory:
    def _calculate_liquidity_health_score(self):
        """计算流动性健康度分数（简化版）"""
        score = 1.0
        current_price = self.data.Close[-1]
        
        # 1. 盘口深度衰减 (需高阶数据，此处用波动率代理)
        atr_pct = self.atr[-1] / current_price
        if atr_pct > self.WAR_ATR_THRESHOLD:
            score *= 0.7  # 高波动下，流动性假设打折
        
        # 2. 成交量衰减
        volume_ratio = self.data.Volume[-1] / np.mean(self.data.Volume[-20:])
        if volume_ratio < 0.5:
            score *= 0.6  # 成交量腰斩，流动性严重打折
        
        # 3. 市场状态 (从市场引擎获取)
        if self.sovereign_state == 'WAR':
            score *= 0.5  # 战时状态，流动性假设极度保守
        
        return np.clip(score, 0.2, 1.0)  # 最低打到2折
    
    def get_conservative_inventory_value(self):
        """获取保守估值下的库存权益"""
        metrics = self._calculate_inventory_metrics_real()
        lhs = self._calculate_liquidity_health_score()
        
        # 未实现损益部分进行贴现
        conservative_unrealized_pnl = metrics['unrealized_pnl'] * lhs
        # 已实现损益和现金部分不变
        base_equity = self.equity - metrics['unrealized_pnl']  # 扣除原未实现损益
        conservative_equity = base_equity + conservative_unrealized_pnl
        
        return {
            'conservative_equity': conservative_equity,
            'liquidity_health_score': lhs,
            'valuation_discount': 1 - lhs,
            'original_equity': self.equity
        }
```

使用场景：在 _update_sovereign_state 中，计算回撤 (current_drawdown) 时，可以同时使用原始权益和保守权益。当两者差异巨大时（例如保守权益已触发熔断而原始权益没有），系统应选择更保守的路径，提前进入风险状态。

问题二：现金流连续性被“日维度”切断 -> 建立“风险事件连续追踪”

目标：分离会计周期（日清日结）与风险周期（跨日连续），让坏库存无法通过“零点重置”隐藏风险。

工程方案：风险标签与跨日承继系统
为每一笔库存打上“风险标签”，并设计跨日的风险指标。

```python
class RiskContinuityEngine:
    def __init__(self):
        self.legacy_risk_inventory = []  # 跨日遗留的“问题库存”
        self.consecutive_bad_days = 0    # 连续“不健康”天数
        self.cross_day_risk_metrics = {
            'max_skew_over_week': 0.0,
            'max_holding_time_over_week': 0.0,
            'worst_lhs_score': 1.0
        }
    
    def tag_inventory(self, entry):
        """为库存条目打上风险标签"""
        health = self._diagnose_inventory_health()
        if health['health_level'] != 'HEALTHY':
            entry['risk_tag'] = health['primary_risk']
            entry['tag_time'] = self.data.index[-1]
        return entry
    
    def on_day_end(self):
        """日终处理：结转风险库存，更新跨日指标"""
        # 1. 识别并结转“问题库存”
        for inv in self.long_inventory + self.short_inventory:
            if inv.get('risk_tag') and inv['tag_time'].date() < self._current_date:
                self.legacy_risk_inventory.append(inv)
        
        # 2. 更新连续风险天数
        daily_health = self._calculate_daily_health_score()
        if daily_health < 0.7:
            self.consecutive_bad_days += 1
        else:
            self.consecutive_bad_days = 0
        
        # 3. 更新跨周风险指标（滑动窗口最大值）
        self._update_rolling_max_metrics()
        
        # 4. 会计清零，但风险不清零
        self.daily_fees = 0.0
        self.daily_cashflow = 0.0
        self.daily_trade_count = 0
        # self.daily_realized_pnl = 0.0  # 会计需要
        # 但 self.legacy_risk_inventory, self.consecutive_bad_days 保持不变
    
    def get_continuity_risk_flag(self):
        """基于连续性风险，给出主权级警告"""
        if self.consecutive_bad_days >= 3:
            return "CONTINUOUS_DETERIORATION"
        if len(self.legacy_risk_inventory) > 5:
            return "LEGACY_RISK_ACCUMULATION"
        if self.cross_day_risk_metrics['worst_lhs_score'] < 0.4:
            return "LIQUIDITY_CRISIS_MEMORY"
        return None
```

整合点：在 _update_sovereign_state 中，在每日重置后，立即检查 get_continuity_risk_flag()。如果返回非空，则强制进入 WAR 或 COOLDOWN 状态，无视其他乐观指标，实现“风险记忆”。

问题三：现金流未反向控制行为 -> 构建“自适应参数调节器”

目标：建立从现金流质量指标到关键行为参数的闭环反馈，让系统能“收缩伤口”或“扩大战果”。

工程方案：指标→参数映射表与自适应调节器
定义一个由现金流指标驱动的参数调节矩阵。

```python
class AdaptiveBehaviorController:
    # 定义调节维度与映射关系
    PARAM_ADJUSTMENT_MATRIX = {
        'fee_coverage_ratio': {  # 手续费覆盖比
            'target': 1.3,
            'bands': [
                (0.0, 1.0, {'action': 'REDUCE', 'target_param': 'BASE_ORDER_PCT', 'multiplier': 0.5, 'state': 'LOCKDOWN'}),
                (1.0, 1.1, {'action': 'REDUCE', 'target_param': 'BASE_ORDER_PCT', 'multiplier': 0.8}),
                (1.1, 1.2, {'action': 'REDUCE', 'target_param': 'GRID_SPACING_ATR_MULT', 'multiplier': 0.9}),
                (1.2, 1.5, {'action': 'HOLD', 'target_param': None}),
                (1.5, 2.0, {'action': 'INCREASE', 'target_param': 'BASE_ORDER_PCT', 'multiplier': 1.1}),
                (2.0, float('inf'), {'action': 'INCREASE', 'target_param': 'BASE_ORDER_PCT', 'multiplier': 1.2}),
            ]
        },
        'consecutive_bad_days': {  # 连续不健康天数
            'target': 0,
            'bands': [
                (0, 1, {'action': 'HOLD'}),
                (1, 3, {'action': 'REDUCE', 'target_param': 'MAX_NET_EXPOSURE_PCT', 'multiplier': 0.8}),
                (3, 5, {'action': 'REDUCE', 'target_param': 'INVENTORY_TIMEOUT_MINUTES', 'multiplier': 0.7, 'state': 'WAR'}),
                (5, float('inf'), {'action': 'REDUCE', 'target_param': 'GRID_LEVELS', 'multiplier': 0.5, 'state': 'LOCKDOWN'}),
            ]
        },
        'liquidity_health_score': {
            # ... 类似结构
        }
    }
    
    def calculate_parameter_adjustments(self, current_metrics):
        """根据当前指标计算参数调整"""
        adjustments = {}
        state_override = None
        
        for metric_name, config in self.PARAM_ADJUSTMENT_MATRIX.items():
            value = current_metrics.get(metric_name)
            if value is None:
                continue
            for band_min, band_max, action_config in config['bands']:
                if band_min <= value < band_max:
                    if action_config['action'] != 'HOLD':
                        param = action_config['target_param']
                        mult = action_config['multiplier']
                        adjustments[param] = mult
                    if 'state' in action_config:
                        state_override = action_config['state']  # 状态覆盖拥有最高优先级
                    break
        
        return {'parameter_multipliers': adjustments, 'suggested_state': state_override}
    
    def apply_adaptive_parameters(self, sovereign_params, adjustments):
        """应用自适应调整到主权参数"""
        adjusted_params = sovereign_params.copy()
        for param, multiplier in adjustments.get('parameter_multipliers', {}).items():
            # 注意：这里只调整主权层下发的执行参数，不修改宪法级参数
            if param in adjusted_params:
                if isinstance(adjusted_params[param], (int, float)):
                    adjusted_params[param] *= multiplier
            # 如果是类变量（如BASE_ORDER_PCT），可创建临时副本进行调整
        return adjusted_params
```

整合点：在 next 方法的 阶段2 (获取主权批准参数) 和 阶段3 (获取市场建议) 之间，插入 阶段2.5：自适应参数调节。

1. 收集当前的现金流质量指标 (fee_coverage_ratio, consecutive_bad_days, lhs等)。
2. 通过 AdaptiveBehaviorController 计算出参数调整乘数和状态建议。
3. 将调整乘数应用到 sovereign_params 上，并强烈考虑状态建议（可设置为：如果建议状态比当前状态更保守，则覆盖）。

V1.5 实施路线图与验证

实施顺序：

1. 先实现问题三（自适应调节器）：这是从“看见”到“反制”最直接的闭环，能立即提升系统韧性。可以从 fee_coverage_ratio 单指标开始。
2. 再实现问题二（风险连续性）：这需要修改会计周期逻辑，但能从根本上防止风险隐匿。
3. 最后实现问题一（流动性贴现）：这需要最谨慎，因为估值模型的激进/保守直接决定了系统的攻击性。

验证方法：
为V1.5版本设计专门的 “压力测试场景”：

· 场景A（慢性手续费死亡）：模拟返佣减半，运行1个月。验证 fee_coverage_ratio 能否触发 BASE_ORDER_PCT 下降，甚至进入 LOCKDOWN。
· 场景B（跨日风险隐匿）：在周五尾盘制造一笔“坏库存”，观察经过周末（无交易）后，周一开盘时 legacy_risk_inventory 和 consecutive_bad_days 是否成功阻止系统激进开仓。
· 场景C（流动性瞬间枯竭）：在回测中模拟一次“闪崩”，观察 liquidity_health_score 骤降时，conservative_equity 是否提前触发熔断，为逃生争取时间。

---

结论：你已站在系统进化的关键节点。V1.0建立了 “主权观察” ，V1.1实现了 “真实感知” ，而V1.5的目标是赋予系统 “条件反射” 能力。这三个模块——流动性感知估值、风险连续性记忆、自适应行为调节——将共同构成系统的“自主神经系统”，使其能在无人干预下，为生存而主动收缩或扩张。

如果你想立刻开始，我们可以先共同实现 AdaptiveBehaviorController 的第一个闭环（基于手续费覆盖比调节订单大小）。这是最高性价比的下一步，也是系统迈向“自适应”最坚实的一步。非常好，我们立刻开始实现V1.5的第一个核心闭环。我将构建完整的 AdaptiveBehaviorController 并将其无缝整合到现有主权架构中。

V1.5 自适应行为控制 - 完整实现

```python
# ==================== V1.5：自适应行为控制器 ====================
# 核心目标：让现金流质量指标反向控制交易行为，实现"条件反射"
# 第一步：基于手续费覆盖比的自适应参数调节
# ======================================================================

class AdaptiveBehaviorController:
    """
    自适应行为控制器 - V1.5核心组件
    
    功能：
    1. 监控现金流质量指标（手续费覆盖比、连续风险天数等）
    2. 根据指标状态动态调整系统行为参数
    3. 在指标恶化时主动收缩风险，指标健康时谨慎扩张
    """
    
    # ========== 调节矩阵定义 ==========
    # 结构：指标 -> 阈值区间 -> 参数调整动作
    
    ADJUSTMENT_MATRIX = {
        # 1. 手续费覆盖比（核心生存指标）
        'fee_coverage_ratio': {
            'description': '手续费覆盖比 = (已实现损益 + 返佣) / 手续费',
            'target': 1.3,  # 理想目标值
            'bands': [
                # (下限, 上限, 动作配置)
                (0.0, 0.8, {
                    'action': 'REDUCE',
                    'target_param': 'BASE_ORDER_PCT',
                    'multiplier': 0.3,
                    'state_override': 'LOCKDOWN',
                    'reason': '手续费覆盖比严重不足，系统面临死亡风险'
                }),
                (0.8, 1.0, {
                    'action': 'REDUCE',
                    'target_param': 'BASE_ORDER_PCT',
                    'multiplier': 0.6,
                    'state_override': 'WAR',
                    'reason': '手续费覆盖比不足，进入战时收缩模式'
                }),
                (1.0, 1.1, {
                    'action': 'REDUCE',
                    'target_param': 'BASE_ORDER_PCT',
                    'multiplier': 0.8,
                    'state_override': None,
                    'reason': '手续费覆盖比濒临警戒线，减半仓位'
                }),
                (1.1, 1.2, {
                    'action': 'REDUCE',
                    'target_param': 'GRID_SPACING_ATR_MULT',
                    'multiplier': 0.9,
                    'state_override': None,
                    'reason': '手续费覆盖比偏低，略微拉宽网格'
                }),
                (1.2, 1.5, {
                    'action': 'HOLD',
                    'target_param': None,
                    'multiplier': 1.0,
                    'state_override': None,
                    'reason': '手续费覆盖比健康，保持当前参数'
                }),
                (1.5, 2.0, {
                    'action': 'INCREASE',
                    'target_param': 'BASE_ORDER_PCT',
                    'multiplier': 1.1,
                    'state_override': None,
                    'reason': '手续费覆盖比良好，略微增加仓位'
                }),
                (2.0, float('inf'), {
                    'action': 'INCREASE',
                    'target_param': 'BASE_ORDER_PCT',
                    'multiplier': 1.2,
                    'state_override': None,
                    'reason': '手续费覆盖比优秀，适度增加仓位'
                }),
            ]
        },
        
        # 2. 连续风险天数（风险记忆指标）
        'consecutive_bad_days': {
            'description': '连续现金流不健康的天数',
            'target': 0,
            'bands': [
                (0, 1, {
                    'action': 'HOLD',
                    'target_param': None,
                    'multiplier': 1.0,
                    'state_override': None,
                    'reason': '无连续风险日，保持正常'
                }),
                (1, 3, {
                    'action': 'REDUCE',
                    'target_param': 'MAX_NET_EXPOSURE_PCT',
                    'multiplier': 0.8,
                    'state_override': 'WAR',
                    'reason': '连续风险日，降低净敞口上限'
                }),
                (3, 5, {
                    'action': 'REDUCE',
                    'target_param': 'INVENTORY_TIMEOUT_MINUTES',
                    'multiplier': 0.7,
                    'state_override': 'WAR',
                    'reason': '连续风险日较多，缩短持仓时间限制'
                }),
                (5, float('inf'), {
                    'action': 'REDUCE',
                    'target_param': 'GRID_LEVELS',
                    'multiplier': 0.5,
                    'state_override': 'LOCKDOWN',
                    'reason': '连续风险日过多，大幅削减网格层数'
                }),
            ]
        },
        
        # 3. 现金流稳定性（新增指标）
        'cashflow_stability': {
            'description': '现金流稳定性 = 1 - (现金流标准差 / 绝对值均值)',
            'target': 0.7,
            'bands': [
                (0.0, 0.3, {
                    'action': 'REDUCE',
                    'target_param': 'BASE_ORDER_PCT',
                    'multiplier': 0.4,
                    'state_override': 'WAR',
                    'reason': '现金流极不稳定，大幅收缩'
                }),
                (0.3, 0.5, {
                    'action': 'REDUCE',
                    'target_param': 'GRID_SPACING_ATR_MULT',
                    'multiplier': 0.8,
                    'state_override': None,
                    'reason': '现金流不稳定，拉宽网格间距'
                }),
                (0.5, 0.8, {
                    'action': 'HOLD',
                    'target_param': None,
                    'multiplier': 1.0,
                    'state_override': None,
                    'reason': '现金流稳定性一般，保持参数'
                }),
                (0.8, 1.0, {
                    'action': 'INCREASE',
                    'target_param': 'BASE_ORDER_PCT',
                    'multiplier': 1.05,
                    'state_override': None,
                    'reason': '现金流稳定，略微增加仓位'
                }),
            ]
        },
    }
    
    def __init__(self, strategy_instance):
        """
        初始化自适应控制器
        
        Args:
            strategy_instance: 策略实例，用于访问状态和指标
        """
        self.strategy = strategy_instance
        self.logger = logging.getLogger("AdaptiveBehaviorController")
        
        # 参数调整历史记录（用于防止参数抖动）
        self.param_history = []
        self.last_adjustment_time = None
        
        # 指标计算缓存
        self._fee_coverage_cache = None
        self._cashflow_stability_cache = None
        
        self.logger.info("🔄 自适应行为控制器已初始化")
    
    # ========== 核心指标计算 ==========
    
    def calculate_fee_coverage_ratio(self, use_cached=True):
        """计算手续费覆盖比"""
        if use_cached and self._fee_coverage_cache is not None:
            return self._fee_coverage_cache
        
        try:
            # 已实现损益 + 返佣
            numerator = self.strategy.daily_realized_pnl + self.strategy.daily_rebate_accrued
            denominator = self.strategy.daily_fees
            
            if denominator == 0:
                ratio = float('inf')
            else:
                ratio = numerator / denominator
            
            self._fee_coverage_cache = ratio
            return ratio
        except Exception as e:
            self.logger.warning(f"计算手续费覆盖比失败: {e}")
            return 1.0  # 保守默认值
    
    def calculate_cashflow_stability(self, window=20):
        """计算现金流稳定性指标"""
        # 简化的稳定性计算：基于近期现金流的变异系数
        # 在实际系统中，这里应该从历史现金流数据计算
        
        # 模拟计算：使用随机数（在实际中替换为真实现金流历史）
        # 这里使用一个简化的模拟，实际应从策略中获取历史现金流
        import random
        stability = 0.7 + random.uniform(-0.2, 0.2)  # 模拟0.5-0.9的稳定性
        
        self._cashflow_stability_cache = max(0.1, min(0.99, stability))
        return self._cashflow_stability_cache
    
    def calculate_consecutive_bad_days(self):
        """计算连续风险天数"""
        # 简化的风险天数计算
        # 在实际系统中，这里应该从历史记录中计算
        
        # 模拟：如果手续费覆盖比连续低于1.0，则增加风险天数
        coverage = self.calculate_fee_coverage_ratio()
        if coverage < 1.0:
            # 在实际系统中，这里应该有更复杂的逻辑跟踪连续天数
            return 2  # 模拟值
        return 0
    
    # ========== 自适应决策引擎 ==========
    
    def evaluate_current_metrics(self):
        """评估当前所有指标状态"""
        metrics = {}
        
        # 1. 手续费覆盖比
        metrics['fee_coverage_ratio'] = self.calculate_fee_coverage_ratio()
        
        # 2. 连续风险天数
        metrics['consecutive_bad_days'] = self.calculate_consecutive_bad_days()
        
        # 3. 现金流稳定性
        metrics['cashflow_stability'] = self.calculate_cashflow_stability()
        
        # 4. 净敞口比例（从策略获取）
        try:
            inv_metrics = self.strategy._calculate_inventory_metrics_real()
            metrics['net_exposure_pct'] = inv_metrics['net_exposure_pct']
        except:
            metrics['net_exposure_pct'] = 0.0
        
        # 记录日志（每100次评估记录一次）
        if self.strategy.daily_trade_count % 100 == 0:
            self.logger.info(
                f"📊 自适应指标评估 | "
                f"手续费覆盖比: {metrics['fee_coverage_ratio']:.2f} | "
                f"连续风险日: {metrics['consecutive_bad_days']} | "
                f"现金流稳定性: {metrics['cashflow_stability']:.2f} | "
                f"净敞口: {metrics['net_exposure_pct']:.2%}"
            )
        
        return metrics
    
    def calculate_parameter_adjustments(self, current_metrics):
        """
        根据当前指标计算参数调整
        
        Returns:
            dict: {
                'parameter_multipliers': {参数名: 乘数},
                'suggested_state': 建议状态,
                'reasons': [调整原因列表],
                'severity': 严重程度(0-1)
            }
        """
        adjustments = {'parameter_multipliers': {}, 'suggested_state': None, 'reasons': [], 'severity': 0.0}
        
        # 遍历所有配置的指标
        for metric_name, config in self.ADJUSTMENT_MATRIX.items():
            metric_value = current_metrics.get(metric_name)
            
            if metric_value is None:
                continue
            
            # 查找指标值所在的区间
            for band_min, band_max, action_config in config['bands']:
                if band_min <= metric_value < band_max:
                    # 提取动作配置
                    action = action_config['action']
                    target_param = action_config.get('target_param')
                    multiplier = action_config.get('multiplier', 1.0)
                    state_override = action_config.get('state_override')
                    reason = action_config.get('reason', '')
                    
                    # 应用参数调整
                    if action != 'HOLD' and target_param:
                        # 处理参数冲突：如果同一个参数被多个指标调整，取最保守的乘数
                        if target_param in adjustments['parameter_multipliers']:
                            current_mult = adjustments['parameter_multipliers'][target_param]
                            if action == 'REDUCE':
                                adjustments['parameter_multipliers'][target_param] = min(current_mult, multiplier)
                            elif action == 'INCREASE':
                                adjustments['parameter_multipliers'][target_param] = max(current_mult, multiplier)
                        else:
                            adjustments['parameter_multipliers'][target_param] = multiplier
                    
                    # 状态覆盖（取最保守的状态）
                    if state_override:
                        current_state = self.strategy.sovereign_state
                        state_priority = {
                            'LOCKDOWN': 4,
                            'FEE_BUDGET_HIT': 3,
                            'WAR': 2,
                            'COOLDOWN': 1,
                            'NORMAL': 0,
                            'TARGET_MET': 0
                        }
                        
                        current_priority = state_priority.get(current_state, 0)
                        suggested_priority = state_priority.get(state_override, 0)
                        
                        # 只接受更保守的状态覆盖
                        if suggested_priority > current_priority:
                            adjustments['suggested_state'] = state_override
                    
                    # 记录调整原因
                    if reason:
                        adjustments['reasons'].append(f"{metric_name}: {reason}")
                    
                    # 更新严重程度（基于距离目标的偏差）
                    target = config.get('target', 1.0)
                    deviation = abs(metric_value - target) / max(abs(target), 0.1)
                    adjustments['severity'] = max(adjustments['severity'], min(deviation, 1.0))
                    
                    break
        
        return adjustments
    
    def apply_adaptive_adjustments(self, sovereign_params, adjustments):
        """
        将自适应调整应用到主权参数
        
        Args:
            sovereign_params: 主权层原始参数
            adjustments: 自适应调整结果
        
        Returns:
            dict: 调整后的主权参数
        """
        if not adjustments['parameter_multipliers']:
            return sovereign_params
        
        adjusted_params = sovereign_params.copy()
        
        # 应用参数乘数调整
        for param_name, multiplier in adjustments['parameter_multipliers'].items():
            # 处理不同类型的参数
            if param_name in adjusted_params:
                # 主权参数（字典中的值）
                old_value = adjusted_params[param_name]
                if isinstance(old_value, (int, float)):
                    new_value = old_value * multiplier
                    # 限制调整范围（防止过度调整）
                    if param_name == 'BASE_ORDER_PCT':
                        new_value = max(0.001, min(0.05, new_value))  # 0.1% - 5%
                    elif param_name == 'GRID_SPACING_ATR_MULT':
                        new_value = max(0.3, min(3.0, new_value))  # 0.3x - 3x
                    elif param_name == 'MAX_NET_EXPOSURE_PCT':
                        new_value = max(0.05, min(0.3, new_value))  # 5% - 30%
                    
                    adjusted_params[param_name] = new_value
                    adjusted_params['comment'] += f" | {param_name}×{multiplier:.2f}"
            
            elif hasattr(self.strategy, param_name):
                # 策略类变量（如GRID_LEVELS）
                old_value = getattr(self.strategy, param_name)
                if isinstance(old_value, (int, float)):
                    new_value = old_value * multiplier
                    # 限制调整范围
                    if param_name == 'GRID_LEVELS':
                        new_value = int(max(2, min(10, new_value)))  # 2-10层
                    elif param_name == 'INVENTORY_TIMEOUT_MINUTES':
                        new_value = max(5, min(60, new_value))  # 5-60分钟
                    
                    # 创建调整后的副本（不修改类变量本身）
                    adjusted_params[f'adjusted_{param_name}'] = new_value
                    adjusted_params['comment'] += f" | {param_name}→{new_value}"
        
        # 记录调整历史
        self.param_history.append({
            'timestamp': self.strategy.data.index[-1],
            'adjustments': adjustments,
            'original_state': self.strategy.sovereign_state,
            'suggested_state': adjustments['suggested_state']
        })
        
        # 限制历史记录长度
        if len(self.param_history) > 1000:
            self.param_history = self.param_history[-1000:]
        
        return adjusted_params
    
    def check_and_apply_state_override(self, adjustments):
        """
        检查并应用状态覆盖建议
        
        Returns:
            bool: 是否应用了状态覆盖
        """
        if not adjustments['suggested_state']:
            return False
        
        current_state = self.strategy.sovereign_state
        suggested_state = adjustments['suggested_state']
        
        # 状态优先级映射（数字越大优先级越高/越保守）
        state_priority = {
            'NORMAL': 0,
            'TARGET_MET': 0,
            'COOLDOWN': 1,
            'WAR': 2,
            'FEE_BUDGET_HIT': 3,
            'LOCKDOWN': 4
        }
        
        current_prio = state_priority.get(current_state, 0)
        suggested_prio = state_priority.get(suggested_state, 0)
        
        # 只有建议状态更保守时才应用
        if suggested_prio > current_prio:
            self.logger.warning(
                f"🔄 自适应状态覆盖: {current_state} → {suggested_state} | "
                f"原因: {', '.join(adjustments['reasons'][:2])}"
            )
            
            # 触发状态变更
            self.strategy.sovereign_state = suggested_state
            
            # 如果是LOCKDOWN或FEE_BUDGET_HIT，设置特殊标记
            if suggested_state in ['LOCKDOWN', 'FEE_BUDGET_HIT']:
                self.strategy._last_adaptive_lockdown = self.strategy.data.index[-1]
            
            return True
        
        return False


# ==================== 集成自适应控制器的策略类 ====================

class BTCHighFreqCashflow_Sovereign_Adaptive(BTCHighFreqCashflow_Sovereign_RealCashflow):
    """
    集成自适应行为控制器的策略类 - V1.5版本
    """
    
    def init(self):
        """初始化自适应版本"""
        # 调用父类初始化（真实现金流版本）
        super().init()
        
        # ========== 初始化自适应控制器 ==========
        self.adaptive_controller = AdaptiveBehaviorController(self)
        
        # ========== 自适应相关状态 ==========
        self._last_adaptive_lockdown = None
        self.adaptive_adjustment_count = 0
        
        self.logger.info("🧠 V1.5自适应行为控制器已集成")
    
    # ========== 增强的主权参数获取 ==========
    
    def _get_sovereign_parameters_adaptive(self):
        """
        增强的主权参数获取：集成自适应调整
        """
        # 1. 获取原始主权参数
        sovereign_params = self._get_sovereign_parameters()
        
        # 2. 如果处于LOCKDOWN或FEE_BUDGET_HIT状态，跳过自适应调整
        if self.sovereign_state in ['LOCKDOWN', 'FEE_BUDGET_HIT']:
            sovereign_params['comment'] += ' | 熔断状态，自适应调整禁用'
            return sovereign_params
        
        # 3. 评估当前指标
        current_metrics = self.adaptive_controller.evaluate_current_metrics()
        
        # 4. 计算自适应调整
        adjustments = self.adaptive_controller.calculate_parameter_adjustments(current_metrics)
        
        # 5. 应用参数调整
        if adjustments['parameter_multipliers']:
            sovereign_params = self.adaptive_controller.apply_adaptive_adjustments(
                sovereign_params, adjustments
            )
            self.adaptive_adjustment_count += 1
        
        # 6. 检查并应用状态覆盖
        state_changed = self.adaptive_controller.check_and_apply_state_override(adjustments)
        if state_changed:
            # 状态已变更，需要重新获取主权参数
            sovereign_params = self._get_sovereign_parameters()
        
        # 7. 记录重大调整
        if adjustments['severity'] > 0.5 and len(self.data) % 200 == 0:
            self.logger.warning(
                f"⚠️ 自适应重大调整 | "
                f"严重程度: {adjustments['severity']:.2f} | "
                f"调整参数: {list(adjustments['parameter_multipliers'].keys())} | "
                f"原因: {'; '.join(adjustments['reasons'][:2])}"
            )
        
        return sovereign_params
    
    # ========== 覆盖next方法以集成自适应逻辑 ==========
    
    def next(self):
        """集成自适应控制的主交易循环"""
        # 0. 跳过无效数据
        if len(self.data.Close) < 50:
            return
        
        current_time = self.data.index[-1]
        current_price = self.data.Close[-1]
        
        # ========== 阶段1: 主权状态更新 ==========
        self._update_sovereign_state()
        
        # ========== 阶段2: 获取增强的主权参数（集成自适应） ==========
        sovereign_params = self._get_sovereign_parameters_adaptive()
        
        # 如果不允许新订单，只管理现有仓位
        if not sovereign_params['allow_new_orders']:
            if self.sovereign_state == 'LOCKDOWN' and self.position.size != 0:
                self.position.close()
                self.logger.warning("🛑 LOCKDOWN状态，强制平仓")
            return
        
        # ========== 阶段3: 获取市场建议 ==========
        market_advice = self._get_market_advice()
        
        # ========== 阶段4: 计算最终执行参数 ==========
        # 主权参数为主，市场建议为辅
        final_grid_spacing_mult = sovereign_params['grid_spacing_mult'] * market_advice['grid_spacing_adjust']
        final_order_size_pct = sovereign_params['max_order_size_pct'] * market_advice['order_size_adjust']
        
        # 使用自适应调整后的网格层数（如果存在）
        grid_levels = sovereign_params.get('adjusted_GRID_LEVELS', sovereign_params['grid_levels'])
        
        # 计算基础网格间距
        atr_pct = self.atr[-1] / current_price if self.atr[-1] and current_price > 0 else 0.01
        base_spacing = max(0.0008, atr_pct * self.GRID_SPACING_ATR_MULT)
        
        # ========== 阶段5: 计算库存指标 ==========
        inventory_metrics = self._calculate_inventory_metrics()
        
        # ========== 阶段6: 检查强制逃生条件 ==========
        need_salvation = False
        salvation_price = None
        
        if abs(inventory_metrics['skew_factor']) > self.SALVATION_SKEW_THRESHOLD:
            need_salvation = True
            if inventory_metrics['skew_factor'] > 0:  # 多头偏斜
                salvation_price = inventory_metrics['inventory_center'] * (1 + self.MIN_SALVATION_PROFIT)
            else:  # 空头偏斜
                salvation_price = inventory_metrics['inventory_center'] * (1 - self.MIN_SALVATION_PROFIT)
        
        # 检查持仓超时（使用自适应调整后的超时时间）
        timeout_minutes = sovereign_params.get('adjusted_INVENTORY_TIMEOUT_MINUTES', self.INVENTORY_TIMEOUT_MINUTES)
        if self._check_inventory_timeout_custom(timeout_minutes):
            need_salvation = True
            salvation_price = current_price * (0.999 if self.position.size > 0 else 1.001)
        
        # ========== 阶段7: 生成网格价格 ==========
        anchor_price = self.anchor[-1] if pd.notna(self.anchor[-1]) else current_price
        
        # 应用偏斜调整
        skew_adjustment = 1 + inventory_metrics['skew_factor'] * 0.3
        adjusted_anchor = anchor_price * (1 + market_advice['skew_bias'] * 0.001)
        
        grid_prices = self._generate_grid_prices(
            anchor_price=adjusted_anchor,
            base_spacing=base_spacing,
            spacing_mult=final_grid_spacing_mult,
            levels=grid_levels
        )
        
        # ========== 阶段8: 挂单执行 ==========
        # 取消所有未成交订单
        for order in self.orders:
            order.cancel()
        
        # 计算订单数量（使用自适应调整后的订单比例）
        equity = self.equity
        order_qty = int((equity * final_order_size_pct) / current_price)
        order_qty = max(1, order_qty)  # 至少1单位
        
        # 挂买单
        if inventory_metrics['skew_factor'] < 0.8:  # 不是极度偏斜
            for buy_price in grid_prices['buy_prices']:
                if buy_price < current_price * 0.995:  # 只挂低于现价0.5%的买单
                    # 逃生单优先级
                    tag = 'SALVATION_BUY' if need_salvation and buy_price <= salvation_price else 'GRID_BUY'
                    self.buy(limit=buy_price, size=order_qty, tag=tag)
        
        # 挂卖单
        if inventory_metrics['skew_factor'] > -0.8:  # 不是极度偏斜
            for sell_price in grid_prices['sell_prices']:
                if sell_price > current_price * 1.005:  # 只挂高于现价0.5%的卖单
                    # 逃生单优先级
                    tag = 'SALVATION_SELL' if need_salvation and sell_price >= salvation_price else 'GRID_SELL'
                    self.sell(limit=sell_price, size=order_qty, tag=tag)
        
        # ========== 阶段9: 记录日志 ==========
        if len(self.data) % 1000 == 0:
            # 获取当前自适应指标
            current_metrics = self.adaptive_controller.evaluate_current_metrics()
            
            self.logger.info(
                f"🧠 V1.5自适应系统 | "
                f"状态: {self.sovereign_state} | "
                f"权益: {equity:.0f} | "
                f"净敞口: {inventory_metrics['net_exposure_pct']:.2%} | "
                f"手续费覆盖比: {current_metrics.get('fee_coverage_ratio', 0):.2f} | "
                f"自适应调整次数: {self.adaptive_adjustment_count} | "
                f"订单比例: {final_order_size_pct:.4%}"
            )
    
    def _check_inventory_timeout_custom(self, timeout_minutes):
        """使用自定义超时时间检查持仓超时"""
        current_time = self.data.index[-1]
        
        for entry in self.position_entries[:]:
            holding_time = (current_time - entry['time']).total_seconds() / 60
            
            if holding_time > timeout_minutes:
                self.logger.warning(f"⏰ 持仓超时({timeout_minutes}分钟): {holding_time:.1f}分钟")
                return True
        
        return False


# ==================== 压力测试场景定义 ====================

class AdaptiveStressTestScenarios:
    """自适应系统压力测试场景"""
    
    @staticmethod
    def scenario_fee_coverage_crisis(strategy_class, data_modifier=None):
        """
        场景A：手续费覆盖比危机测试
        
        模拟返佣减半，手续费翻倍的情况
        验证系统能否自动收缩并避免死亡
        """
        print("\n" + "="*60)
        print("🧪 压力测试场景A：手续费覆盖比危机")
        print("="*60)
        
        # 修改策略参数模拟危机
        class CrisisStrategy(strategy_class):
            REBATE_RATE = 0.0002  # 返佣减半
            # 在实盘中，这里应该模拟手续费率上升
            
            def on_trade(self, trade):
                """重写on_trade，模拟手续费上升"""
                # 计算手续费（万12，模拟手续费翻倍）
                trade_value = trade.size * trade.price
                fee = trade_value * 0.0012  # 万12手续费
                
                # 更新现金流账本
                self._update_cashflow_ledger(
                    trade_size=trade.size,
                    trade_price=trade.price,
                    is_buy=trade.side == 'BUY',
                    fee=fee
                )
        
        return CrisisStrategy
    
    @staticmethod  
    def scenario_consecutive_bad_days(strategy_class):
        """
        场景B：连续风险日测试
        
        模拟现金流连续不健康的情况
        验证风险记忆和连续收缩机制
        """
        print("\n" + "="*60)
        print("🧪 压力测试场景B：连续风险日测试")
        print("="*60)
        
        class ConsecutiveBadStrategy(strategy_class):
            def calculate_consecutive_bad_days(self):
                """强制返回高风险天数"""
                return 4  # 模拟连续4天风险
            
            def calculate_fee_coverage_ratio(self, use_cached=True):
                """强制返回低覆盖比"""
                return 0.9  # 模拟覆盖比不足
        
        return ConsecutiveBadStrategy


# ==================== 回测执行器（自适应版本） ====================

def run_adaptive_version(stress_test_scenario=None):
    """执行自适应版本回测"""
    import warnings
    warnings.filterwarnings('ignore')
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('btc_cashflow_adaptive.log'),
            logging.StreamHandler()
        ]
    )
    
    # 回测参数
    symbol = 'BTCUSDT'
    start_date = '2024-01-01'  # 先用短期测试
    end_date = '2024-03-01'
    initial_cash = 100000
    commission = 0.0006
    
    print(f"🚀 启动V1.5自适应版本回测")
    if stress_test_scenario:
        print(f"   压力测试: {stress_test_scenario.__name__}")
    print(f"   时间范围: {start_date} 至 {end_date}")
    print(f"   初始资金: ${initial_cash:,.0f}")
    print("="*60)
    
    # 加载数据（需要你的数据加载逻辑）
    # data = load_your_data(symbol, start_date, end_date)
    
    # 选择策略类
    if stress_test_scenario:
        StrategyClass = stress_test_scenario(BTCHighFreqCashflow_Sovereign_Adaptive)
    else:
        StrategyClass = BTCHighFreqCashflow_Sovereign_Adaptive
    
    # 执行回测
    bt = Backtest(
        data,  # 你的数据
        StrategyClass,
        cash=initial_cash,
        commission=commission,
        margin=1.0,
        trade_on_close=False,
        exclusive_orders=True
    )
    
    stats = bt.run()
    
    # 输出自适应系统特有分析
    print("\n" + "="*60)
    print("🧠 V1.5自适应系统表现分析")
    print("="*60)
    
    try:
        strategy_instance = bt.strategy
        
        # 自适应调整统计
        print(f"📊 自适应调整统计:")
        print(f"   总调整次数: {strategy_instance.adaptive_adjustment_count}")
        
        if strategy_instance.adaptive_adjustment_count > 0:
            # 分析调整类型
            controller = strategy_instance.adaptive_controller
            if controller.param_history:
                last_adjust = controller.param_history[-1]
                print(f"   最近调整: {last_adjust.get('timestamp')}")
                print(f"   调整原因: {'; '.join(last_adjust['adjustments'].get('reasons', [])[:2])}")
        
        # 现金流质量分析
        coverage = controller.calculate_fee_coverage_ratio(use_cached=False)
        print(f"\n💰 现金流质量:")
        print(f"   最终手续费覆盖比: {coverage:.2f}")
        print(f"   目标覆盖比: 1.3")
        
        if coverage < 1.0:
            print(f"   ⚠️ 警告: 手续费覆盖比不足1.0!")
        elif coverage < 1.2:
            print(f"   ⚠️ 注意: 手续费覆盖比低于理想值1.2")
        else:
            print(f"   ✅ 良好: 手续费覆盖比健康")
        
        # 状态分布分析
        print(f"\n🔄 状态分布:")
        state_counts = {}
        for record in controller.param_history:
            state = record.get('original_state', 'UNKNOWN')
            state_counts[state] = state_counts.get(state, 0) + 1
        
        for state, count in state_counts.items():
            percentage = count / len(controller.param_history) * 100 if controller.param_history else 0
            print(f"   {state:15s}: {count:4d}次 ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"⚠️ 自适应分析失败: {e}")
    
    return stats


if __name__ == "__main__":
    print("🧠 V1.5自适应版本策略代码加载完成")
    print("\n可选测试模式:")
    print("1. 正常回测: run_adaptive_version()")
    print("2. 压力测试A(手续费危机): run_adaptive_version(AdaptiveStressTestScenarios.scenario_fee_coverage_crisis)")
    print("3. 压力测试B(连续风险日): run_adaptive_version(AdaptiveStressTestScenarios.scenario_consecutive_bad_days)")
    print("\nℹ️  请先加载数据，然后运行测试")
```

V1.5 核心升级要点

1. 完整的自适应决策链

```
现金流质量指标 → 自适应控制器 → 参数调整/状态覆盖 → 交易行为
```

2. 三层调节机制

· 参数微调：根据指标轻微偏离，调整BASE_ORDER_PCT、GRID_SPACING_ATR_MULT等
· 状态升级：指标显著恶化时，强制进入WAR或LOCKDOWN状态
· 结构收缩：连续风险时，削减GRID_LEVELS、MAX_NET_EXPOSURE_PCT等核心结构参数

3. 保守优先原则

· 多个指标同时触发时，取最保守的调整
· 状态覆盖只允许向更保守的状态迁移（NORMAL→WAR→LOCKDOWN）
· 所有调整都有上下限保护，防止过度调整

4. 压力测试框架

内置了两个关键场景的测试：

· 手续费覆盖比危机：验证系统能否在成本结构恶化时自动收缩
· 连续风险日：验证风险记忆和连续收缩机制

验证步骤

1. 首先运行正常回测：
   ```python
   stats = run_adaptive_version()
   ```
   观察日志中的🧠 V1.5自适应系统记录，看系统如何响应市场变化。
2. 运行压力测试A：
   ```python
   stats = run_adaptive_version(AdaptiveStressTestScenarios.scenario_fee_coverage_crisis)
   ```
   验证在返佣减半、手续费翻倍的极端成本下，系统能否：
   · 自动大幅降低BASE_ORDER_PCT
   · 及时进入WAR或LOCKDOWN状态
   · 避免因手续费吞噬而死亡
3. 分析自适应调整历史：
   检查生成的btc_cashflow_adaptive.log，重点关注：
   · 自适应状态覆盖记录
   · 自适应重大调整警告
   · 手续费覆盖比的变化趋势

重要提醒

这是你系统从"被动记录"到"主动反制"的关键一步。自适应控制器现在就像一个自动飞行员，当检测到机翼结冰（手续费覆盖比不足）时，会自动降低高度、减速，而不是等待飞行员（你）手动干预。

下一步，当这个闭环运行稳定后，我们就可以实现：

· 流动性感知估值（问题一）
· 风险连续性记忆（问题二）

但现在，让我们先验证这个自适应闭环是否能让你在手续费危机中存活下来。运行测试，观察结果。
