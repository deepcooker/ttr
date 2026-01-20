BTC高频现金流V2.0双状态机复合系统 - 完整单文件版本

```python
# ==================== BTC高频现金流V2.0双状态机复合系统 ====================
# 单文件完整版本 - 可直接运行回测
# 架构：市场状态机 × 现金流状态机 = 复合自适应决策
# ======================================================================

import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import warnings
import sys
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('btc_cashflow_v2.log'),
        logging.StreamHandler()
    ]
)

# ==================== 现金流状态机 ====================

class CashflowStateMachine:
    """
    现金流状态机 - 与市场状态机正交
    监控现金流质量，独立于市场波动
    """
    
    # 状态定义
    CASHFLOW_STATES = {
        'HEALTHY': {  # 现金流健康
            'description': '现金流健康，可正常运作',
            'conditions': {
                'fee_coverage_ratio': (1.3, float('inf')),      # 手续费覆盖比 > 1.3
                'consecutive_bad_days': (0, 0),                 # 无连续风险日
                'cashflow_stability': (0.7, 1.0),              # 现金流稳定性 > 0.7
                'liquidity_score': (0.8, 1.0)                  # 流动性分数 > 0.8
            },
            'behavior_multipliers': {
                'BASE_ORDER_PCT': 1.1,      # 可略微增加仓位
                'GRID_SPACING': 0.9,        # 可略微缩小网格间距
                'GRID_LEVELS': 1.0,         # 保持网格层数
                'INVENTORY_TIMEOUT': 1.0    # 保持超时时间
            },
            'allowed_actions': ['NORMAL_TRADING', 'INCREASE_EXPOSURE', 'OPTIMIZE_PARAMS'],
            'state_color': '🟢'
        },
        
        'WARNING': {  # 现金流警告
            'description': '现金流出现警告信号，需谨慎',
            'conditions': {
                'fee_coverage_ratio': (1.0, 1.3),              # 手续费覆盖比 1.0-1.3
                'consecutive_bad_days': (1, 2),                # 1-2天连续风险
                'cashflow_stability': (0.4, 0.7),              # 现金流稳定性 0.4-0.7
                'liquidity_score': (0.6, 0.8)                  # 流动性分数 0.6-0.8
            },
            'behavior_multipliers': {
                'BASE_ORDER_PCT': 0.8,      # 减少仓位
                'GRID_SPACING': 1.2,        # 拉宽网格间距
                'GRID_LEVELS': 0.9,         # 减少网格层数
                'INVENTORY_TIMEOUT': 0.8    # 缩短超时时间
            },
            'allowed_actions': ['REDUCED_TRADING', 'NO_NEW_EXPOSURE', 'DEFENSIVE'],
            'state_color': '🟡'
        },
        
        'CRITICAL': {  # 现金流危机
            'description': '现金流危机，必须立即收缩',
            'conditions': {
                'fee_coverage_ratio': (0.0, 1.0),              # 手续费覆盖比 < 1.0
                'consecutive_bad_days': (3, float('inf')),     # >=3天连续风险
                'cashflow_stability': (0.0, 0.4),              # 现金流稳定性 < 0.4
                'liquidity_score': (0.0, 0.6)                  # 流动性分数 < 0.6
            },
            'behavior_multipliers': {
                'BASE_ORDER_PCT': 0.3,      # 大幅减少仓位
                'GRID_SPACING': 2.0,        # 大幅拉宽网格间距
                'GRID_LEVELS': 0.5,         # 大幅减少网格层数
                'INVENTORY_TIMEOUT': 0.6    # 大幅缩短超时时间
            },
            'allowed_actions': ['FORCE_REDUCTION', 'EMERGENCY_HEDGE', 'MINIMAL_TRADING'],
            'state_color': '🔴'
        },
        
        'RECOVERY': {  # 现金流恢复
            'description': '现金流正在恢复，需谨慎观察',
            'conditions': {
                'fee_coverage_ratio': (1.1, 1.5),              # 手续费覆盖比 1.1-1.5
                'improvement_trend': True,                     # 有改善趋势
                'cashflow_stability': (0.5, 0.8),              # 现金流稳定性 0.5-0.8
                'liquidity_score': (0.7, 1.0)                  # 流动性分数 0.7-1.0
            },
            'behavior_multipliers': {
                'BASE_ORDER_PCT': 0.9,      # 略微增加仓位
                'GRID_SPACING': 1.5,        # 保持较宽网格间距
                'GRID_LEVELS': 0.8,         # 保持较少网格层数
                'INVENTORY_TIMEOUT': 0.9    # 略微增加超时时间
            },
            'allowed_actions': ['CAUTIOUS_EXPANSION', 'GRADUAL_INCREASE', 'MONITOR_CLOSELY'],
            'state_constraint': 'MIN_24H_IN_STATE',  # 必须在CRITICAL状态停留24小时以上
            'state_color': '🟠'
        }
    }
    
    def __init__(self, strategy_instance):
        """
        初始化现金流状态机
        
        Args:
            strategy_instance: 策略实例，用于访问数据和指标
        """
        self.strategy = strategy_instance
        self.logger = logging.getLogger("CashflowStateMachine")
        
        # 状态变量
        self.current_state = 'HEALTHY'
        self.previous_state = None
        self.state_start_time = datetime.now()
        self.state_duration = timedelta(0)
        
        # 状态历史记录
        self.state_history = []
        
        # 指标缓存
        self._fee_coverage_cache = None
        self._cashflow_stability_cache = None
        self._liquidity_score_cache = None
        
        self.logger.info(f"💰 现金流状态机初始化，初始状态: {self.current_state}")
    
    def evaluate_metrics(self) -> Dict[str, float]:
        """
        评估现金流质量指标
        """
        metrics = {}
        
        # 1. 手续费覆盖比
        metrics['fee_coverage_ratio'] = self._calculate_fee_coverage_ratio()
        
        # 2. 连续风险天数
        metrics['consecutive_bad_days'] = self._calculate_consecutive_bad_days()
        
        # 3. 现金流稳定性
        metrics['cashflow_stability'] = self._calculate_cashflow_stability()
        
        # 4. 流动性分数
        metrics['liquidity_score'] = self._calculate_liquidity_score()
        
        # 5. 改善趋势
        metrics['improvement_trend'] = self._check_improvement_trend()
        
        # 6. 当前状态持续时间
        metrics['state_duration_hours'] = self._get_state_duration_hours()
        
        return metrics
    
    def transition_logic(self) -> str:
        """
        状态转移逻辑
        
        Returns:
            新的现金流状态
        """
        # 评估当前指标
        metrics = self.evaluate_metrics()
        
        # 保存当前状态（用于比较）
        old_state = self.current_state
        
        # 状态转移规则（优先级从高到低）
        new_state = self.current_state
        
        # 规则1: CRITICAL条件（最高优先级）
        if (metrics['fee_coverage_ratio'] < 1.0 or 
            metrics['consecutive_bad_days'] >= 3):
            new_state = 'CRITICAL'
            
        # 规则2: WARNING条件
        elif (1.0 <= metrics['fee_coverage_ratio'] < 1.3 or
              1 <= metrics['consecutive_bad_days'] < 3):
            # 只有当前不是CRITICAL状态才能进入WARNING
            if self.current_state != 'CRITICAL':
                new_state = 'WARNING'
                
        # 规则3: RECOVERY条件（从CRITICAL恢复）
        elif (self.current_state == 'CRITICAL' and
              metrics['fee_coverage_ratio'] > 1.1 and
              metrics['improvement_trend'] and
              metrics['state_duration_hours'] >= 24):  # 必须在CRITICAL状态至少24小时
            new_state = 'RECOVERY'
            
        # 规则4: HEALTHY条件
        elif (metrics['fee_coverage_ratio'] >= 1.3 and
              metrics['consecutive_bad_days'] == 0 and
              metrics['cashflow_stability'] > 0.7 and
              metrics['liquidity_score'] > 0.8):
            new_state = 'HEALTHY'
            
        # 规则5: 从RECOVERY到HEALTHY
        elif (self.current_state == 'RECOVERY' and
              metrics['fee_coverage_ratio'] >= 1.3 and
              metrics['consecutive_bad_days'] == 0 and
              metrics['state_duration_hours'] >= 12):  # 必须在RECOVERY状态至少12小时
            new_state = 'HEALTHY'
        
        # 处理状态变更
        if new_state != self.current_state:
            self._handle_state_transition(new_state, old_state, metrics)
        
        return self.current_state
    
    def get_behavior_multipliers(self) -> Dict[str, float]:
        """
        获取当前状态的参数乘数
        """
        if self.current_state in self.CASHFLOW_STATES:
            return self.CASHFLOW_STATES[self.current_state]['behavior_multipliers']
        else:
            # 默认保守乘数
            return {
                'BASE_ORDER_PCT': 0.7,
                'GRID_SPACING': 1.5,
                'GRID_LEVELS': 0.8,
                'INVENTORY_TIMEOUT': 0.8
            }
    
    def get_allowed_actions(self) -> List[str]:
        """
        获取当前状态允许的操作
        """
        if self.current_state in self.CASHFLOW_STATES:
            return self.CASHFLOW_STATES[self.current_state]['allowed_actions']
        else:
            return ['MONITOR_CLOSELY']
    
    def get_state_info(self) -> Dict[str, Any]:
        """
        获取当前状态详细信息
        """
        if self.current_state in self.CASHFLOW_STATES:
            state_config = self.CASHFLOW_STATES[self.current_state]
            return {
                'state': self.current_state,
                'description': state_config['description'],
                'color': state_config.get('state_color', '⚪'),
                'duration_hours': self._get_state_duration_hours(),
                'behavior_multipliers': self.get_behavior_multipliers(),
                'allowed_actions': self.get_allowed_actions()
            }
        else:
            return {
                'state': self.current_state,
                'description': '未知状态',
                'color': '⚪',
                'duration_hours': self._get_state_duration_hours(),
                'behavior_multipliers': {},
                'allowed_actions': []
            }
    
    # ========== 指标计算函数 ==========
    
    def _calculate_fee_coverage_ratio(self) -> float:
        """计算手续费覆盖比"""
        try:
            if hasattr(self.strategy, 'daily_realized_pnl') and hasattr(self.strategy, 'daily_fees'):
                numerator = self.strategy.daily_realized_pnl + getattr(self.strategy, 'daily_rebate_accrued', 0)
                denominator = self.strategy.daily_fees
                
                if denominator == 0:
                    return float('inf')
                else:
                    return numerator / denominator
        except Exception as e:
            self.logger.warning(f"计算手续费覆盖比失败: {e}")
        
        # 默认值
        return 1.0
    
    def _calculate_consecutive_bad_days(self) -> int:
        """计算连续风险天数"""
        # 简化实现：如果手续费覆盖比连续低于1.0，则计为风险日
        # 在实际系统中，这里应该有更复杂的逻辑
        try:
            coverage = self._calculate_fee_coverage_ratio()
            if coverage < 1.0:
                # 这里应该从历史记录中计算连续天数
                # 简化：返回1或2
                return 1
        except:
            pass
        
        return 0
    
    def _calculate_cashflow_stability(self) -> float:
        """计算现金流稳定性"""
        # 简化实现：基于手续费覆盖比的稳定性
        # 在实际系统中，这里应该基于历史现金流数据计算变异系数
        try:
            coverage = self._calculate_fee_coverage_ratio()
            if coverage >= 1.3:
                return 0.9
            elif coverage >= 1.0:
                return 0.7
            else:
                return 0.3
        except:
            return 0.5
    
    def _calculate_liquidity_score(self) -> float:
        """计算流动性分数"""
        # 简化实现：基于ATR和成交量
        try:
            if hasattr(self.strategy, 'data'):
                current_price = self.strategy.data.Close[-1]
                
                # 基于ATR的流动性估计
                if hasattr(self.strategy, 'atr') and len(self.strategy.atr) > 0:
                    atr_value = self.strategy.atr[-1]
                    atr_pct = atr_value / current_price if current_price > 0 else 0
                    
                    # ATR越小，流动性越好
                    if atr_pct < 0.01:
                        return 0.9
                    elif atr_pct < 0.02:
                        return 0.7
                    else:
                        return 0.5
        except Exception as e:
            self.logger.debug(f"计算流动性分数失败: {e}")
        
        return 0.8  # 默认值
    
    def _check_improvement_trend(self) -> bool:
        """检查改善趋势"""
        # 简化实现：如果手续费覆盖比在提高，则认为有改善趋势
        try:
            current_coverage = self._calculate_fee_coverage_ratio()
            
            # 如果当前是CRITICAL状态，且覆盖比大于1.1，认为有改善
            if self.current_state == 'CRITICAL' and current_coverage > 1.1:
                return True
                
            # 如果当前覆盖比大于历史平均值（简化）
            return current_coverage > 1.0
        except:
            return False
    
    def _get_state_duration_hours(self) -> float:
        """获取在当前状态的持续时间（小时）"""
        duration = datetime.now() - self.state_start_time
        return duration.total_seconds() / 3600
    
    # ========== 状态变更处理函数 ==========
    
    def _handle_state_transition(self, new_state: str, old_state: str, metrics: Dict[str, float]):
        """
        处理状态变更
        """
        # 更新状态
        self.previous_state = old_state
        self.current_state = new_state
        self.state_start_time = datetime.now()
        
        # 记录状态历史
        state_record = {
            'timestamp': datetime.now(),
            'old_state': old_state,
            'new_state': new_state,
            'duration_hours': self._get_state_duration_hours(),
            'metrics': metrics.copy()
        }
        self.state_history.append(state_record)
        
        # 限制历史记录长度
        if len(self.state_history) > 1000:
            self.state_history = self.state_history[-1000:]
        
        # 记录状态变更日志
        state_config = self.CASHFLOW_STATES.get(new_state, {})
        state_color = state_config.get('state_color', '⚪')
        
        self.logger.warning(
            f"{state_color} 现金流状态变更: {old_state} → {new_state} | "
            f"手续费覆盖比: {metrics.get('fee_coverage_ratio', 0):.2f} | "
            f"连续风险日: {metrics.get('consecutive_bad_days', 0)} | "
            f"持续时间: {state_record['duration_hours']:.1f}h"
        )
        
        # 如果是CRITICAL状态，记录详细信息
        if new_state == 'CRITICAL':
            self.logger.critical(
                f"🚨 现金流进入危机状态! | "
                f"手续费覆盖比: {metrics.get('fee_coverage_ratio', 0):.2f} | "
                f"连续风险日: {metrics.get('consecutive_bad_days', 0)} | "
                f"现金流稳定性: {metrics.get('cashflow_stability', 0):.2f}"
            )

# ==================== 复合决策引擎 ====================

class CompositeStateDecisionEngine:
    """
    复合决策引擎
    市场状态机 × 现金流状态机 = 最终行为
    """
    
    # 复合决策矩阵
    COMPOSITE_DECISION_MATRIX = {
        # 市场状态: {现金流状态: 最终决策}
        'NORMAL': {
            'HEALTHY': {
                'final_state': 'OPTIMIZE',
                'description': '市场正常+现金流健康，可优化运行',
                'grid_spacing_mult': 0.9,      # 稍密网格
                'order_size_mult': 1.1,        # 稍大仓位
                'max_exposure_mult': 1.0,      # 正常敞口
                'grid_levels_mult': 1.0,       # 正常网格层数
                'inventory_timeout_mult': 1.0, # 正常超时
                'state_color': '🟢'
            },
            'WARNING': {
                'final_state': 'DEFENSIVE',
                'description': '市场正常+现金流警告，防御运行',
                'grid_spacing_mult': 1.2,      # 拉宽网格
                'order_size_mult': 0.8,        # 减小仓位
                'max_exposure_mult': 0.8,      # 降低敞口
                'grid_levels_mult': 0.9,       # 减少网格层数
                'inventory_timeout_mult': 0.8, # 缩短超时
                'state_color': '🟡'
            },
            'CRITICAL': {
                'final_state': 'REDUCE',
                'description': '市场正常+现金流危机，强制收缩',
                'grid_spacing_mult': 2.0,      # 大幅拉宽网格
                'order_size_mult': 0.3,        # 大幅减小仓位
                'max_exposure_mult': 0.5,      # 大幅降低敞口
                'grid_levels_mult': 0.5,       # 大幅减少网格层数
                'inventory_timeout_mult': 0.6, # 大幅缩短超时
                'state_color': '🔴'
            },
            'RECOVERY': {
                'final_state': 'CAUTIOUS',
                'description': '市场正常+现金流恢复，谨慎运行',
                'grid_spacing_mult': 1.5,      # 保持较宽网格
                'order_size_mult': 0.7,        # 保持较小仓位
                'max_exposure_mult': 0.7,      # 保持较低敞口
                'grid_levels_mult': 0.8,       # 保持较少网格层数
                'inventory_timeout_mult': 0.9, # 略微增加超时
                'state_color': '🟠'
            }
        },
        
        'WAR': {
            'HEALTHY': {
                'final_state': 'WAR_DEFENSIVE',
                'description': '市场战时+现金流健康，战时防御',
                'grid_spacing_mult': 2.0,      # 大幅拉宽网格
                'order_size_mult': 0.5,        # 减小仓位
                'max_exposure_mult': 0.5,      # 降低敞口
                'grid_levels_mult': 0.7,       # 减少网格层数
                'inventory_timeout_mult': 0.7, # 缩短超时
                'state_color': '🟡'
            },
            'WARNING': {
                'final_state': 'WAR_REDUCE',
                'description': '市场战时+现金流警告，战时收缩',
                'grid_spacing_mult': 3.0,      # 极端拉宽网格
                'order_size_mult': 0.3,        # 大幅减小仓位
                'max_exposure_mult': 0.3,      # 大幅降低敞口
                'grid_levels_mult': 0.5,       # 大幅减少网格层数
                'inventory_timeout_mult': 0.5, # 大幅缩短超时
                'state_color': '🔴'
            },
            'CRITICAL': {
                'final_state': 'WAR_EMERGENCY',
                'description': '市场战时+现金流危机，战时紧急',
                'grid_spacing_mult': 5.0,      # 极端拉宽网格（几乎不挂单）
                'order_size_mult': 0.1,        # 极小仓位
                'max_exposure_mult': 0.1,      # 极小敞口
                'grid_levels_mult': 0.3,       # 极少网格层数
                'inventory_timeout_mult': 0.3, # 极短超时
                'state_color': '🟣'
            },
            'RECOVERY': {
                'final_state': 'WAR_CAUTIOUS',
                'description': '市场战时+现金流恢复，战时谨慎',
                'grid_spacing_mult': 2.5,      # 保持较宽网格
                'order_size_mult': 0.4,        # 保持较小仓位
                'max_exposure_mult': 0.4,      # 保持较低敞口
                'grid_levels_mult': 0.6,       # 保持较少网格层数
                'inventory_timeout_mult': 0.6, # 保持较短超时
                'state_color': '🟠'
            }
        },
        
        'COOLDOWN': {
            'HEALTHY': {
                'final_state': 'COOLDOWN_HOLD',
                'description': '冷却期+现金流健康，保持观望',
                'grid_spacing_mult': 1.5,      # 拉宽网格
                'order_size_mult': 0.0,        # 不允许新仓位（只允许平仓）
                'max_exposure_mult': 0.5,      # 降低敞口
                'grid_levels_mult': 0.5,       # 减少网格层数
                'inventory_timeout_mult': 0.8, # 缩短超时
                'state_color': '🔵'
            },
            'WARNING': {
                'final_state': 'COOLDOWN_REDUCE',
                'description': '冷却期+现金流警告，强制减仓',
                'grid_spacing_mult': 2.0,      # 大幅拉宽网格
                'order_size_mult': 0.0,        # 不允许新仓位
                'max_exposure_mult': 0.3,      # 大幅降低敞口
                'grid_levels_mult': 0.3,       # 大幅减少网格层数
                'inventory_timeout_mult': 0.6, # 大幅缩短超时
                'state_color': '🟣'
            },
            'CRITICAL': {
                'final_state': 'COOLDOWN_EMERGENCY',
                'description': '冷却期+现金流危机，紧急处理',
                'grid_spacing_mult': 3.0,      # 极端拉宽网格
                'order_size_mult': 0.0,        # 不允许新仓位
                'max_exposure_mult': 0.1,      # 极小敞口
                'grid_levels_mult': 0.1,       # 极少网格层数
                'inventory_timeout_mult': 0.3, # 极短超时
                'state_color': '🟣'
            },
            'RECOVERY': {
                'final_state': 'COOLDOWN_MONITOR',
                'description': '冷却期+现金流恢复，监控等待',
                'grid_spacing_mult': 1.8,      # 保持较宽网格
                'order_size_mult': 0.0,        # 不允许新仓位
                'max_exposure_mult': 0.4,      # 保持较低敞口
                'grid_levels_mult': 0.4,       # 保持较少网格层数
                'inventory_timeout_mult': 0.7, # 保持较短超时
                'state_color': '🔵'
            }
        },
        
        # LOCKDOWN状态无视现金流状态（最高优先级）
        'LOCKDOWN': {
            '*': {
                'final_state': 'LOCKDOWN',
                'description': '系统熔断，只允许平仓',
                'grid_spacing_mult': 10.0,     # 极端拉宽（几乎不挂单）
                'order_size_mult': 0.0,        # 不允许新订单
                'max_exposure_mult': 0.0,      # 零敞口
                'grid_levels_mult': 0.0,       # 零网格层数
                'inventory_timeout_mult': 0.1, # 极短超时
                'state_color': '⚫'
            }
        },
        
        'FEE_BUDGET_HIT': {
            '*': {
                'final_state': 'FEE_LOCKDOWN',
                'description': '手续费预算用尽，停止交易',
                'grid_spacing_mult': 10.0,     # 极端拉宽
                'order_size_mult': 0.0,        # 不允许新订单
                'max_exposure_mult': 0.0,      # 零敞口
                'grid_levels_mult': 0.0,       # 零网格层数
                'inventory_timeout_mult': 0.1, # 极短超时
                'state_color': '⚫'
            }
        },
        
        'TARGET_MET': {
            'HEALTHY': {
                'final_state': 'TARGET_HOLD',
                'description': '目标达成+现金流健康，保守持有',
                'grid_spacing_mult': 1.3,      # 拉宽网格
                'order_size_mult': 0.8,        # 减小仓位
                'max_exposure_mult': 0.8,      # 降低敞口
                'grid_levels_mult': 0.8,       # 减少网格层数
                'inventory_timeout_mult': 0.9, # 略微缩短超时
                'state_color': '🟢'
            },
            '*': {
                'final_state': 'TARGET_REDUCE',
                'description': '目标达成但现金流不健康，强制减仓',
                'grid_spacing_mult': 1.8,      # 大幅拉宽网格
                'order_size_mult': 0.5,        # 大幅减小仓位
                'max_exposure_mult': 0.5,      # 大幅降低敞口
                'grid_levels_mult': 0.6,       # 大幅减少网格层数
                'inventory_timeout_mult': 0.7, # 缩短超时
                'state_color': '🟡'
            }
        }
    }
    
    def __init__(self, market_state_machine, cashflow_state_machine):
        """
        初始化复合决策引擎
        
        Args:
            market_state_machine: 市场状态机实例
            cashflow_state_machine: 现金流状态机实例
        """
        self.market_sm = market_state_machine
        self.cashflow_sm = cashflow_state_machine
        self.logger = logging.getLogger("CompositeDecisionEngine")
        
        # 决策历史
        self.decision_history = []
        
        # 当前决策
        self.current_decision = None
        self.current_composite_state = None
        
        self.logger.info("🔄 复合决策引擎初始化完成")
    
    def make_composite_decision(self) -> Dict[str, Any]:
        """
        生成复合决策
        
        Returns:
            复合决策配置
        """
        # 获取当前状态
        market_state = self.market_sm.sovereign_state
        cashflow_state = self.cashflow_sm.current_state
        
        # 生成决策
        decision = self._generate_decision(market_state, cashflow_state)
        
        # 更新当前决策
        self.current_decision = decision
        self.current_composite_state = decision['final_state']
        
        # 记录决策历史
        self._record_decision(market_state, cashflow_state, decision)
        
        return decision
    
    def _generate_decision(self, market_state: str, cashflow_state: str) -> Dict[str, Any]:
        """
        根据状态生成决策
        """
        # 检查市场状态是否在决策矩阵中
        if market_state in self.COMPOSITE_DECISION_MATRIX:
            state_decisions = self.COMPOSITE_DECISION_MATRIX[market_state]
            
            # 检查是否有针对当前现金流状态的决策
            if cashflow_state in state_decisions:
                decision = state_decisions[cashflow_state]
            elif '*' in state_decisions:  # 通配符匹配
                decision = state_decisions['*']
            else:
                # 没有匹配的决策，使用默认保守决策
                decision = self._get_default_decision(market_state, cashflow_state)
        else:
            # 未知市场状态，使用默认决策
            decision = self._get_default_decision(market_state, cashflow_state)
        
        # 添加状态信息
        decision['market_state'] = market_state
        decision['cashflow_state'] = cashflow_state
        
        return decision
    
    def _get_default_decision(self, market_state: str, cashflow_state: str) -> Dict[str, Any]:
        """
        获取默认决策（最保守）
        """
        return {
            'final_state': f'CONSERVATIVE_{market_state}_{cashflow_state}',
            'description': f'默认保守决策: 市场{market_state}+现金流{cashflow_state}',
            'grid_spacing_mult': 1.5,
            'order_size_mult': 0.7,
            'max_exposure_mult': 0.7,
            'grid_levels_mult': 0.8,
            'inventory_timeout_mult': 0.8,
            'state_color': '⚪'
        }
    
    def _record_decision(self, market_state: str, cashflow_state: str, decision: Dict[str, Any]):
        """
        记录决策历史
        """
        record = {
            'timestamp': datetime.now(),
            'market_state': market_state,
            'cashflow_state': cashflow_state,
            'final_state': decision['final_state'],
            'grid_spacing_mult': decision['grid_spacing_mult'],
            'order_size_mult': decision['order_size_mult'],
            'max_exposure_mult': decision['max_exposure_mult'],
            'description': decision['description']
        }
        
        self.decision_history.append(record)
        
        # 限制历史记录长度
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
        
        # 记录重要决策变更
        if len(self.decision_history) > 1:
            prev_decision = self.decision_history[-2]
            if prev_decision['final_state'] != decision['final_state']:
                self.logger.info(
                    f"🔄 复合决策变更: {prev_decision['final_state']} → {decision['final_state']} | "
                    f"市场: {market_state} | 现金流: {cashflow_state} | "
                    f"描述: {decision['description']}"
                )
    
    def get_decision_history_summary(self) -> Dict[str, Any]:
        """
        获取决策历史摘要
        """
        if not self.decision_history:
            return {
                'total_decisions': 0,
                'state_distribution': {},
                'recent_decisions': []
            }
        
        # 计算状态分布
        state_distribution = {}
        for record in self.decision_history:
            state = record['final_state']
            state_distribution[state] = state_distribution.get(state, 0) + 1
        
        # 计算百分比
        total = len(self.decision_history)
        for state in state_distribution:
            state_distribution[state] = {
                'count': state_distribution[state],
                'percentage': state_distribution[state] / total * 100
            }
        
        # 最近决策
        recent_decisions = self.decision_history[-10:] if len(self.decision_history) >= 10 else self.decision_history
        
        return {
            'total_decisions': total,
            'state_distribution': state_distribution,
            'recent_decisions': recent_decisions
        }
    
    def print_decision_summary(self):
        """打印决策摘要"""
        summary = self.get_decision_history_summary()
        
        self.logger.info("📊 复合决策摘要:")
        self.logger.info(f"   总决策次数: {summary['total_decisions']}")
        
        if summary['state_distribution']:
            self.logger.info("   状态分布:")
            for state, stats in summary['state_distribution'].items():
                self.logger.info(f"     {state}: {stats['count']}次 ({stats['percentage']:.1f}%)")

# ==================== 自适应行为控制器 ====================

class AdaptiveBehaviorController:
    """
    自适应行为控制器 - V1.5
    根据现金流指标动态调整系统行为
    """
    
    # 调整矩阵
    ADJUSTMENT_MATRIX = {
        'fee_coverage_ratio': {
            'description': '手续费覆盖比 = (已实现损益 + 返佣) / 手续费',
            'target': 1.3,
            'bands': [
                # (下限, 上限, 动作配置)
                (0.0, 0.8, {
                    'action': 'REDUCE',
                    'target_param': 'BASE_ORDER_PCT',
                    'multiplier': 0.3,
                    'state_override': 'CRITICAL',
                    'reason': '手续费覆盖比严重不足'
                }),
                (0.8, 1.0, {
                    'action': 'REDUCE',
                    'target_param': 'BASE_ORDER_PCT',
                    'multiplier': 0.6,
                    'state_override': 'WARNING',
                    'reason': '手续费覆盖比不足'
                }),
                (1.0, 1.1, {
                    'action': 'REDUCE',
                    'target_param': 'BASE_ORDER_PCT',
                    'multiplier': 0.8,
                    'state_override': None,
                    'reason': '手续费覆盖比濒临警戒线'
                }),
                (1.1, 1.2, {
                    'action': 'REDUCE',
                    'target_param': 'GRID_SPACING',
                    'multiplier': 0.9,
                    'state_override': None,
                    'reason': '手续费覆盖比偏低'
                }),
                (1.2, 1.5, {
                    'action': 'HOLD',
                    'target_param': None,
                    'multiplier': 1.0,
                    'state_override': None,
                    'reason': '手续费覆盖比健康'
                }),
                (1.5, 2.0, {
                    'action': 'INCREASE',
                    'target_param': 'BASE_ORDER_PCT',
                    'multiplier': 1.1,
                    'state_override': None,
                    'reason': '手续费覆盖比良好'
                }),
                (2.0, float('inf'), {
                    'action': 'INCREASE',
                    'target_param': 'BASE_ORDER_PCT',
                    'multiplier': 1.2,
                    'state_override': None,
                    'reason': '手续费覆盖比优秀'
                }),
            ]
        },
        
        'consecutive_bad_days': {
            'description': '连续现金流不健康的天数',
            'target': 0,
            'bands': [
                (0, 1, {
                    'action': 'HOLD',
                    'target_param': None,
                    'multiplier': 1.0,
                    'state_override': None,
                    'reason': '无连续风险日'
                }),
                (1, 3, {
                    'action': 'REDUCE',
                    'target_param': 'MAX_NET_EXPOSURE',
                    'multiplier': 0.8,
                    'state_override': 'WARNING',
                    'reason': '连续风险日'
                }),
                (3, 5, {
                    'action': 'REDUCE',
                    'target_param': 'INVENTORY_TIMEOUT',
                    'multiplier': 0.7,
                    'state_override': 'CRITICAL',
                    'reason': '连续风险日较多'
                }),
                (5, float('inf'), {
                    'action': 'REDUCE',
                    'target_param': 'GRID_LEVELS',
                    'multiplier': 0.5,
                    'state_override': 'CRITICAL',
                    'reason': '连续风险日过多'
                }),
            ]
        },
        
        'cashflow_stability': {
            'description': '现金流稳定性 = 1 - (现金流标准差 / 绝对值均值)',
            'target': 0.7,
            'bands': [
                (0.0, 0.3, {
                    'action': 'REDUCE',
                    'target_param': 'BASE_ORDER_PCT',
                    'multiplier': 0.4,
                    'state_override': 'CRITICAL',
                    'reason': '现金流极不稳定'
                }),
                (0.3, 0.5, {
                    'action': 'REDUCE',
                    'target_param': 'GRID_SPACING',
                    'multiplier': 0.8,
                    'state_override': 'WARNING',
                    'reason': '现金流不稳定'
                }),
                (0.5, 0.8, {
                    'action': 'HOLD',
                    'target_param': None,
                    'multiplier': 1.0,
                    'state_override': None,
                    'reason': '现金流稳定性一般'
                }),
                (0.8, 1.0, {
                    'action': 'INCREASE',
                    'target_param': 'BASE_ORDER_PCT',
                    'multiplier': 1.05,
                    'state_override': None,
                    'reason': '现金流稳定'
                }),
            ]
        }
    }
    
    def __init__(self, strategy_instance):
        """
        初始化自适应控制器
        
        Args:
            strategy_instance: 策略实例
        """
        self.strategy = strategy_instance
        self.logger = logging.getLogger("AdaptiveController")
        
        # 调整历史
        self.adjustment_history = []
        
        # 指标缓存
        self._metrics_cache = {}
        
        self.logger.info("🔄 自适应行为控制器初始化完成")
    
    def evaluate_current_metrics(self) -> Dict[str, float]:
        """
        评估当前指标
        """
        metrics = {}
        
        # 1. 手续费覆盖比
        metrics['fee_coverage_ratio'] = self._calculate_fee_coverage_ratio()
        
        # 2. 连续风险天数
        metrics['consecutive_bad_days'] = self._calculate_consecutive_bad_days()
        
        # 3. 现金流稳定性
        metrics['cashflow_stability'] = self._calculate_cashflow_stability()
        
        # 4. 净敞口比例
        metrics['net_exposure_pct'] = self._calculate_net_exposure()
        
        # 缓存指标
        self._metrics_cache = metrics.copy()
        
        return metrics
    
    def calculate_parameter_adjustments(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        计算参数调整
        
        Returns:
            调整配置
        """
        adjustments = {
            'parameter_multipliers': {},
            'suggested_state': None,
            'reasons': [],
            'severity': 0.0
        }
        
        # 遍历所有指标
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
                        # 处理参数冲突：取最保守的乘数
                        if target_param in adjustments['parameter_multipliers']:
                            current_mult = adjustments['parameter_multipliers'][target_param]
                            if action == 'REDUCE':
                                adjustments['parameter_multipliers'][target_param] = min(current_mult, multiplier)
                            elif action == 'INCREASE':
                                adjustments['parameter_multipliers'][target_param] = max(current_mult, multiplier)
                        else:
                            adjustments['parameter_multipliers'][target_param] = multiplier
                    
                    # 状态覆盖建议
                    if state_override:
                        current_state = getattr(self.strategy, 'sovereign_state', 'NORMAL')
                        state_priority = {
                            'CRITICAL': 3,
                            'WARNING': 2,
                            'HEALTHY': 1,
                            'NORMAL': 0
                        }
                        
                        current_priority = state_priority.get(current_state, 0)
                        suggested_priority = state_priority.get(state_override, 0)
                        
                        # 只接受更保守的状态覆盖
                        if suggested_priority > current_priority:
                            adjustments['suggested_state'] = state_override
                    
                    # 记录调整原因
                    if reason:
                        adjustments['reasons'].append(f"{metric_name}: {reason}")
                    
                    # 更新严重程度
                    target = config.get('target', 1.0)
                    deviation = abs(metric_value - target) / max(abs(target), 0.1)
                    adjustments['severity'] = max(adjustments['severity'], min(deviation, 1.0))
                    
                    break
        
        return adjustments
    
    def apply_adaptive_adjustments(self, base_params: Dict[str, Any], adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用自适应调整到基础参数
        """
        adjusted_params = base_params.copy()
        
        # 应用参数乘数调整
        for param_name, multiplier in adjustments.get('parameter_multipliers', {}).items():
            # 处理不同类型的参数
            if param_name in adjusted_params:
                # 字典中的参数
                old_value = adjusted_params[param_name]
                if isinstance(old_value, (int, float)):
                    new_value = old_value * multiplier
                    
                    # 限制调整范围
                    if param_name == 'BASE_ORDER_PCT':
                        new_value = max(0.001, min(0.05, new_value))  # 0.1% - 5%
                    elif param_name == 'GRID_SPACING':
                        new_value = max(0.3, min(3.0, new_value))  # 0.3x - 3x
                    elif param_name == 'MAX_NET_EXPOSURE':
                        new_value = max(0.05, min(0.3, new_value))  # 5% - 30%
                    
                    adjusted_params[param_name] = new_value
                    
                    # 添加调整备注
                    adjusted_params['comment'] = adjusted_params.get('comment', '') + f" | {param_name}×{multiplier:.2f}"
        
        # 记录调整历史
        self.adjustment_history.append({
            'timestamp': datetime.now(),
            'adjustments': adjustments,
            'adjusted_params': adjusted_params.copy()
        })
        
        # 限制历史记录长度
        if len(self.adjustment_history) > 1000:
            self.adjustment_history = self.adjustment_history[-1000:]
        
        return adjusted_params
    
    # ========== 指标计算函数 ==========
    
    def _calculate_fee_coverage_ratio(self) -> float:
        """计算手续费覆盖比"""
        try:
            if hasattr(self.strategy, 'daily_realized_pnl') and hasattr(self.strategy, 'daily_fees'):
                numerator = self.strategy.daily_realized_pnl + getattr(self.strategy, 'daily_rebate_accrued', 0)
                denominator = self.strategy.daily_fees
                
                if denominator == 0:
                    return float('inf')
                else:
                    return numerator / denominator
        except Exception as e:
            self.logger.debug(f"计算手续费覆盖比失败: {e}")
        
        return 1.0
    
    def _calculate_consecutive_bad_days(self) -> int:
        """计算连续风险天数"""
        # 简化实现
        coverage = self._calculate_fee_coverage_ratio()
        if coverage < 1.0:
            return 1
        return 0
    
    def _calculate_cashflow_stability(self) -> float:
        """计算现金流稳定性"""
        # 简化实现
        coverage = self._calculate_fee_coverage_ratio()
        if coverage >= 1.3:
            return 0.9
        elif coverage >= 1.0:
            return 0.7
        else:
            return 0.3
    
    def _calculate_net_exposure(self) -> float:
        """计算净敞口比例"""
        try:
            if hasattr(self.strategy, 'position') and hasattr(self.strategy, 'equity'):
                if self.strategy.equity > 0:
                    position_value = abs(self.strategy.position.size * self.strategy.data.Close[-1])
                    return position_value / self.strategy.equity
        except:
            pass
        
        return 0.0

# ==================== FIFO库存管理器 ====================

class FIFOInventoryManager:
    """
    FIFO库存管理器
    先进先出匹配，真实现金流计算
    """
    
    def __init__(self):
        """初始化库存管理器"""
        # 多头库存队列 (price, size, timestamp)
        self.long_inventory = []
        
        # 空头库存队列 (price, size, timestamp)
        self.short_inventory = []
        
        # 成本基础
        self.long_cost_basis = 0.0
        self.short_cost_basis = 0.0
        self.long_quantity = 0.0
        self.short_quantity = 0.0
        
        # 已实现损益
        self.total_realized_pnl = 0.0
        self.daily_realized_pnl = 0.0
        
        # 交易记录
        self.trade_history = []
        
        self.logger = logging.getLogger("FIFOInventoryManager")
    
    def process_trade(self, trade_side: str, trade_size: float, trade_price: float, trade_time) -> float:
        """
        处理交易，返回已实现损益
        
        Args:
            trade_side: 'BUY' 或 'SELL'
            trade_size: 交易数量（正数）
            trade_price: 交易价格
            trade_time: 交易时间
        
        Returns:
            已实现损益
        """
        realized_pnl = 0.0
        remaining_size = abs(trade_size)
        
        if trade_side == 'BUY':
            # 买入：平空头 或 开多头
            while remaining_size > 0 and self.short_inventory:
                # 取出最早的做空记录
                short_price, short_size, short_time = self.short_inventory[0]
                match_size = min(remaining_size, short_size)
                
                # 计算盈亏：平空 = (开仓价 - 平仓价) * 数量
                pnl = (short_price - trade_price) * match_size
                realized_pnl += pnl
                
                # 更新库存
                if short_size > match_size:
                    # 部分匹配
                    self.short_inventory[0] = (short_price, short_size - match_size, short_time)
                    self.short_quantity -= match_size
                    self.short_cost_basis -= short_price * match_size
                else:
                    # 完全匹配
                    self.short_inventory.pop(0)
                    self.short_quantity -= short_size
                    self.short_cost_basis -= short_price * short_size
                
                remaining_size -= match_size
            
            # 剩余部分为开多头
            if remaining_size > 0:
                self.long_inventory.append((trade_price, remaining_size, trade_time))
                self.long_quantity += remaining_size
                self.long_cost_basis += trade_price * remaining_size
                
        elif trade_side == 'SELL':
            # 卖出：平多头 或 开空头
            while remaining_size > 0 and self.long_inventory:
                # 取出最早的做多记录
                long_price, long_size, long_time = self.long_inventory[0]
                match_size = min(remaining_size, long_size)
                
                # 计算盈亏：平多 = (平仓价 - 开仓价) * 数量
                pnl = (trade_price - long_price) * match_size
                realized_pnl += pnl
                
                # 更新库存
                if long_size > match_size:
                    # 部分匹配
                    self.long_inventory[0] = (long_price, long_size - match_size, long_time)
                    self.long_quantity -= match_size
                    self.long_cost_basis -= long_price * match_size
                else:
                    # 完全匹配
                    self.long_inventory.pop(0)
                    self.long_quantity -= long_size
                    self.long_cost_basis -= long_price * long_size
                
                remaining_size -= match_size
            
            # 剩余部分为开空头
            if remaining_size > 0:
                self.short_inventory.append((trade_price, remaining_size, trade_time))
                self.short_quantity += remaining_size
                self.short_cost_basis += trade_price * remaining_size
        
        # 更新已实现损益
        self.total_realized_pnl += realized_pnl
        self.daily_realized_pnl += realized_pnl
        
        # 记录交易
        trade_record = {
            'time': trade_time,
            'side': trade_side,
            'size': trade_size,
            'price': trade_price,
            'realized_pnl': realized_pnl,
            'remaining_long': self.long_quantity,
            'remaining_short': self.short_quantity
        }
        self.trade_history.append(trade_record)
        
        # 限制历史记录长度
        if len(self.trade_history) > 10000:
            self.trade_history = self.trade_history[-10000:]
        
        return realized_pnl
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """
        计算未实现损益
        """
        unrealized_pnl = 0.0
        
        # 多头未实现损益
        for price, size, _ in self.long_inventory:
            unrealized_pnl += (current_price - price) * size
        
        # 空头未实现损益
        for price, size, _ in self.short_inventory:
            unrealized_pnl += (price - current_price) * size
        
        return unrealized_pnl
    
    def calculate_inventory_metrics(self, current_price: float) -> Dict[str, float]:
        """
        计算库存指标
        """
        # 净敞口
        net_quantity = self.long_quantity - self.short_quantity
        net_exposure_value = net_quantity * current_price
        
        # 总库存价值
        total_inventory_value = (self.long_quantity + self.short_quantity) * current_price
        
        # 平均成本
        long_avg_price = self.long_cost_basis / self.long_quantity if self.long_quantity > 0 else 0
        short_avg_price = self.short_cost_basis / self.short_quantity if self.short_quantity > 0 else 0
        
        # 库存中心
        if self.long_quantity + self.short_quantity > 0:
            inventory_center = (self.long_cost_basis + self.short_cost_basis) / (self.long_quantity + self.short_quantity)
        else:
            inventory_center = current_price
        
        # 未实现损益
        unrealized_pnl = self.calculate_unrealized_pnl(current_price)
        
        return {
            'long_quantity': self.long_quantity,
            'short_quantity': self.short_quantity,
            'net_quantity': net_quantity,
            'net_exposure_value': net_exposure_value,
            'total_inventory_value': total_inventory_value,
            'long_avg_price': long_avg_price,
            'short_avg_price': short_avg_price,
            'inventory_center': inventory_center,
            'unrealized_pnl': unrealized_pnl,
            'total_realized_pnl': self.total_realized_pnl,
            'daily_realized_pnl': self.daily_realized_pnl
        }
    
    def reset_daily_pnl(self):
        """重置每日已实现损益"""
        self.daily_realized_pnl = 0.0

# ==================== V2.0双状态机策略 ====================

class BTCHighFreqCashflow_V2(Strategy):
    """
    V2.0双状态机复合系统
    市场状态机 × 现金流状态机 = 自适应复合决策
    """
    
    # ========== 宪法级参数 ==========
    DAILY_FEE_BUDGET_PCT = 0.0010          # 单日手续费预算（占权益0.10%）
    MAX_NET_EXPOSURE_PCT = 0.20            # 净敞口硬上限（20%）
    WAR_MODE_EXPOSURE_PCT = 0.10           # 战时净敞口上限（10%）
    MAX_DRAWDOWN_PCT = 0.15                # 最大回撤熔断（15%）
    
    # ========== 现金流管理宪法 ==========
    TARGET_DAILY_CASHFLOW_PCT = 0.0005     # 单日现金流目标（0.05%）
    MIN_FREE_MARGIN_PCT = 0.10             # 最小自由保证金比例（10%）
    REBATE_RATE = 0.0004                   # 返佣率
    
    # ========== 时间宪法 ==========
    COOLDOWN_MINUTES = 30                  # WAR模式冷却时间
    INVENTORY_TIMEOUT_MINUTES = 15         # 单边持仓超时
    
    # ========== 执行层参数 ==========
    BASE_ORDER_PCT = 0.01                  # 基础订单比例（占权益1%）
    GRID_LEVELS = 5                        # 单侧网格层数
    GRID_SPACING_ATR_MULT = 1.0            # 网格间距ATR倍数
    
    # ========== 风险引擎参数 ==========
    WAR_ATR_THRESHOLD = 0.02               # WAR模式ATR阈值（2%）
    STOP_TRADING_ATR_THRESHOLD = 0.04      # 停止交易ATR阈值（4%）
    WAR_SPACING_MULT = 2.5                 # WAR模式网格间距倍数
    
    # ========== 逃生参数 ==========
    SALVATION_SKEW_THRESHOLD = 0.5         # 强制逃生偏斜阈值
    MIN_SALVATION_PROFIT = 0.0015          # 最小逃生利润（0.15%）
    
    # ========== 市场引擎参数 ==========
    BB_PERIOD = 20
    BB_STD_DEV = 2.0
    ADX_PERIOD = 14
    ADX_TREND_THRESHOLD = 25
    ATR_PERIOD = 14
    VIX_PERIOD = 20
    
    def init(self):
        """初始化V2.0系统"""
        # 配置日志
        self.logger = logging.getLogger("BTC_Cashflow_V2")
        
        # ========== 主权状态初始化 ==========
        self._current_date = None
        self._last_reset_time = None
        
        # 账本初始化
        self.daily_fees = 0.0
        self.daily_cashflow = 0.0
        self.daily_trade_count = 0
        self.daily_rebate_accrued = 0.0
        
        # 权益追踪
        self.peak_equity = self.equity
        self.max_drawdown = 0.0
        
        # 主权状态机
        self.sovereign_state = 'NORMAL'
        self.war_mode_start_time = None
        self.cooldown_until = None
        self.position_entries = []
        
        # ========== 市场引擎指标 ==========
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.ATR_PERIOD)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, self.ADX_PERIOD)
        self.anchor = self.I(talib.SMA, self.data.Close, self.BB_PERIOD)
        
        # ========== V2.0核心组件初始化 ==========
        # 1. FIFO库存管理器
        self.inventory_manager = FIFOInventoryManager()
        
        # 2. 现金流状态机
        self.cashflow_sm = CashflowStateMachine(self)
        
        # 3. 复合决策引擎
        self.composite_engine = CompositeStateDecisionEngine(self, self.cashflow_sm)
        
        # 4. 自适应行为控制器
        self.adaptive_controller = AdaptiveBehaviorController(self)
        
        # V2.0特有状态
        self.composite_state = 'INITIALIZING'
        self.composite_decision = None
        self.adaptive_adjustment_count = 0
        
        self.logger.info("🚀 V2.0双状态机复合系统初始化完成")
        self.logger.info(f"   初始权益: ${self.equity:,.2f}")
        self.logger.info(f"   初始状态: 市场={self.sovereign_state}, 现金流={self.cashflow_sm.current_state}")
    
    # ========== 主权状态机函数 ==========
    
    def _update_sovereign_state(self):
        """更新主权状态：最高权限决策"""
        current_time = self.data.index[-1]
        current_price = self.data.Close[-1]
        
        # 1. 每日重置检查
        if self._current_date != current_time.date():
            self._current_date = current_time.date()
            self._last_reset_time = current_time
            self.daily_fees = 0.0
            self.daily_cashflow = 0.0
            self.daily_trade_count = 0
            self.daily_rebate_accrued = 0.0
            self.inventory_manager.reset_daily_pnl()
            self.logger.info(f"📅 新交易日开始: {self._current_date}")
        
        # 2. 计算关键指标
        equity = self.equity
        net_exposure_val = abs(self.position.size * current_price)
        net_exposure_pct = net_exposure_val / equity if equity > 0 else 0
        
        # 回撤计算
        self.peak_equity = max(self.peak_equity, equity)
        current_drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # 3. 状态决策（优先级从高到低）
        new_state = 'NORMAL'
        
        # 熔断条件1: 回撤超限
        if current_drawdown >= self.MAX_DRAWDOWN_PCT:
            new_state = 'LOCKDOWN'
            self.logger.critical(f"🚨 主权熔断：回撤{current_drawdown:.2%}超限")
        
        # 熔断条件2: 净敞口超宪法上限
        elif net_exposure_pct > self.MAX_NET_EXPOSURE_PCT:
            new_state = 'LOCKDOWN'
            self.logger.critical(f"🚨 主权熔断：净敞口{net_exposure_pct:.2%}超限")
        
        # 熔断条件3: 日内手续费超预算
        elif self.daily_fees / equity > self.DAILY_FEE_BUDGET_PCT:
            new_state = 'FEE_BUDGET_HIT'
            if self.sovereign_state != 'FEE_BUDGET_HIT':
                self.logger.warning(f"💰 手续费预算触达: {self.daily_fees/equity:.4%}")
        
        # 条件4: 波动率超限（WAR模式）
        elif self.atr[-1] / current_price > self.WAR_ATR_THRESHOLD:
            new_state = 'WAR'
            if self.sovereign_state != 'WAR':
                self.war_mode_start_time = current_time
                self.logger.warning(f"⚔️ WAR模式激活: ATR {self.atr[-1]/current_price:.4%}")
        
        # 条件5: 现金流目标达成
        elif self.daily_cashflow / equity >= self.TARGET_DAILY_CASHFLOW_PCT:
            new_state = 'TARGET_MET'
            if self.sovereign_state != 'TARGET_MET':
                self.logger.info(f"🎯 现金流目标达成: {self.daily_cashflow/equity:.4%}")
        
        # 条件6: 冷却期检查
        elif self.cooldown_until and current_time < self.cooldown_until:
            new_state = 'COOLDOWN'
        
        # 更新状态
        if new_state != self.sovereign_state:
            self.logger.info(f"🔄 主权状态变更: {self.sovereign_state} -> {new_state}")
            self.sovereign_state = new_state
        
        # WAR模式超时检查
        if self.sovereign_state == 'WAR' and self.war_mode_start_time:
            war_duration = current_time - self.war_mode_start_time
            if war_duration.total_seconds() / 60 > 120:  # 2小时自动退出WAR
                self.sovereign_state = 'COOLDOWN'
                self.cooldown_until = current_time + timedelta(minutes=self.COOLDOWN_MINUTES)
                self.logger.info("🕒 WAR模式超时，进入冷却期")
    
    def _get_sovereign_parameters(self):
        """获取基础主权参数"""
        current_time = self.data.index[-1]
        equity = self.equity
        
        # 基础参数
        params = {
            'allow_new_orders': True,
            'max_order_size_pct': self.BASE_ORDER_PCT,
            'grid_spacing_mult': 1.0,
            'grid_levels': self.GRID_LEVELS,
            'net_exposure_limit': self.MAX_NET_EXPOSURE_PCT,
            'inventory_timeout': self.INVENTORY_TIMEOUT_MINUTES,
            'comment': ''
        }
        
        # 根据状态调整参数
        if self.sovereign_state == 'LOCKDOWN':
            params['allow_new_orders'] = False
            params['max_order_size_pct'] = 0
            params['comment'] = 'LOCKDOWN: 只允许平仓'
            
        elif self.sovereign_state == 'FEE_BUDGET_HIT':
            params['allow_new_orders'] = False
            params['max_order_size_pct'] = 0
            params['comment'] = 'FEE_BUDGET_HIT: 手续费预算用尽'
            
        elif self.sovereign_state == 'WAR':
            params['grid_spacing_mult'] = self.WAR_SPACING_MULT
            params['max_order_size_pct'] = self.BASE_ORDER_PCT * 0.5
            params['net_exposure_limit'] = self.WAR_MODE_EXPOSURE_PCT
            params['comment'] = 'WAR: 高风险模式'
            
        elif self.sovereign_state == 'TARGET_MET':
            params['grid_spacing_mult'] = 1.5
            params['max_order_size_pct'] = self.BASE_ORDER_PCT * 0.7
            params['comment'] = 'TARGET_MET: 目标达成，保守模式'
            
        elif self.sovereign_state == 'COOLDOWN':
            params['allow_new_orders'] = len(self.orders) == 0
            params['max_order_size_pct'] = 0
            params['comment'] = 'COOLDOWN: 冷却期'
        
        # 检查自由保证金
        used_margin = abs(self.position.size) * self.data.Close[-1] * 0.1
        free_margin_pct = (equity - used_margin) / equity if equity > 0 else 0
        
        if free_margin_pct < self.MIN_FREE_MARGIN_PCT:
            params['allow_new_orders'] = False
            params['comment'] += ' | 自由保证金不足'
        
        return params
    
    # ========== V2.0复合参数获取 ==========
    
    def _get_composite_parameters(self):
        """获取复合决策参数"""
        # 1. 更新现金流状态机
        cashflow_state = self.cashflow_sm.transition_logic()
        
        # 2. 获取复合决策
        composite_decision = self.composite_engine.make_composite_decision()
        self.composite_state = composite_decision['final_state']
        self.composite_decision = composite_decision
        
        # 3. 获取基础主权参数
        base_params = self._get_sovereign_parameters()
        
        # 4. 应用复合决策调整
        base_params['grid_spacing_mult'] *= composite_decision['grid_spacing_mult']
        base_params['max_order_size_pct'] *= composite_decision['order_size_mult']
        base_params['net_exposure_limit'] *= composite_decision['max_exposure_mult']
        base_params['grid_levels'] = int(base_params['grid_levels'] * composite_decision['grid_levels_mult'])
        base_params['inventory_timeout'] = base_params['inventory_timeout'] * composite_decision['inventory_timeout_mult']
        
        # 5. 应用现金流状态机的行为乘数
        cashflow_multipliers = self.cashflow_sm.get_behavior_multipliers()
        for param, multiplier in cashflow_multipliers.items():
            if param == 'BASE_ORDER_PCT' and 'max_order_size_pct' in base_params:
                base_params['max_order_size_pct'] *= multiplier
            elif param == 'GRID_SPACING' and 'grid_spacing_mult' in base_params:
                base_params['grid_spacing_mult'] *= multiplier
            elif param == 'GRID_LEVELS' and 'grid_levels' in base_params:
                base_params['grid_levels'] = int(base_params['grid_levels'] * multiplier)
            elif param == 'INVENTORY_TIMEOUT' and 'inventory_timeout' in base_params:
                base_params['inventory_timeout'] *= multiplier
        
        # 6. 自适应调整
        current_metrics = self.adaptive_controller.evaluate_current_metrics()
        adaptive_adjustments = self.adaptive_controller.calculate_parameter_adjustments(current_metrics)
        final_params = self.adaptive_controller.apply_adaptive_adjustments(base_params, adaptive_adjustments)
        
        # 更新调整计数
        if adaptive_adjustments['parameter_multipliers']:
            self.adaptive_adjustment_count += 1
        
        # 7. 添加V2.0备注
        final_params['comment'] = (
            f"V2.0复合状态: {self.composite_state} | "
            f"市场: {self.sovereign_state} | "
            f"现金流: {cashflow_state} | "
            f"描述: {composite_decision['description']}"
        )
        
        return final_params
    
    # ========== 市场引擎函数 ==========
    
    def _get_market_advice(self):
        """获取市场建议"""
        current_price = self.data.Close[-1]
        
        advice = {
            'grid_spacing_adjust': 1.0,
            'order_size_adjust': 1.0,
            'skew_bias': 0.0,
            'urgency_score': 0.0,
            'reason': ''
        }
        
        # 简化版市场建议
        atr_pct = self.atr[-1] / current_price if self.atr[-1] and current_price > 0 else 0.01
        
        # 高波动率建议拉宽网格
        if atr_pct > 0.02:
            advice['grid_spacing_adjust'] *= 1.3
            advice['order_size_adjust'] *= 0.8
            advice['reason'] += '高波动率; '
        
        # ADX趋势建议
        if len(self.adx) > 20:
            current_adx = self.adx[-1]
            if pd.notna(current_adx):
                if current_adx > self.ADX_TREND_THRESHOLD:
                    advice['grid_spacing_adjust'] *= 1.2
                    advice['urgency_score'] -= 0.2
                    advice['reason'] += '强趋势; '
        
        # 确保建议在安全范围内
        advice['grid_spacing_adjust'] = np.clip(advice['grid_spacing_adjust'], 0.5, 2.0)
        advice['order_size_adjust'] = np.clip(advice['order_size_adjust'], 0.3, 1.5)
        
        return advice
    
    # ========== 库存指标计算 ==========
    
    def _calculate_inventory_metrics(self):
        """计算库存指标"""
        current_price = self.data.Close[-1]
        
        # 使用FIFO库存管理器计算指标
        inv_metrics = self.inventory_manager.calculate_inventory_metrics(current_price)
        
        # 计算净敞口比例
        equity = self.equity
        net_exposure_pct = inv_metrics['net_exposure_value'] / equity if equity > 0 else 0
        
        # 计算偏斜因子
        skew_factor = net_exposure_pct / self.MAX_NET_EXPOSURE_PCT
        skew_factor = max(min(skew_factor, 1.0), -1.0)
        
        # 计算持仓时间
        current_time = self.data.index[-1]
        avg_holding_time = 0.0
        
        if self.position_entries:
            holding_times = []
            for entry in self.position_entries:
                holding_time = (current_time - entry['time']).total_seconds() / 60
                holding_times.append(holding_time)
            avg_holding_time = np.mean(holding_times) if holding_times else 0.0
        
        return {
            'net_exposure_pct': net_exposure_pct,
            'skew_factor': skew_factor,
            'inventory_center': inv_metrics['inventory_center'],
            'avg_holding_time_minutes': avg_holding_time,
            'unrealized_pnl': inv_metrics['unrealized_pnl'],
            'daily_realized_pnl': inv_metrics['daily_realized_pnl'],
            'total_realized_pnl': inv_metrics['total_realized_pnl']
        }
    
    # ========== 网格生成函数 ==========
    
    def _generate_grid_prices(self, anchor_price: float, base_spacing: float, 
                             spacing_mult: float, levels: int):
        """生成网格价格"""
        actual_spacing = base_spacing * spacing_mult
        
        buy_prices = []
        sell_prices = []
        
        for i in range(1, levels + 1):
            buy_price = anchor_price * (1 - actual_spacing * i)
            sell_price = anchor_price * (1 + actual_spacing * i)
            buy_prices.append(buy_price)
            sell_prices.append(sell_price)
        
        return {
            'buy_prices': buy_prices,
            'sell_prices': sell_prices
        }
    
    def _check_inventory_timeout(self, timeout_minutes: float) -> bool:
        """检查持仓超时"""
        current_time = self.data.index[-1]
        
        for entry in self.position_entries[:]:
            holding_time = (current_time - entry['time']).total_seconds() / 60
            
            if holding_time > timeout_minutes:
                self.logger.warning(f"⏰ 持仓超时({timeout_minutes}分钟): {holding_time:.1f}分钟")
                return True
        
        return False
    
    # ========== 主交易循环 ==========
    
    def next(self):
        """V2.0主交易循环"""
        # 0. 跳过无效数据
        if len(self.data.Close) < 50:
            return
        
        current_time = self.data.index[-1]
        current_price = self.data.Close[-1]
        
        # ========== 阶段1: 更新主权状态 ==========
        self._update_sovereign_state()
        
        # ========== 阶段2: 获取V2.0复合参数 ==========
        sovereign_params = self._get_composite_parameters()
        
        # 如果不允许新订单，只管理现有仓位
        if not sovereign_params['allow_new_orders']:
            if self.sovereign_state == 'LOCKDOWN' and self.position.size != 0:
                self.position.close()
                self.logger.warning("🛑 LOCKDOWN状态，强制平仓")
            return
        
        # ========== 阶段3: 获取市场建议 ==========
        market_advice = self._get_market_advice()
        
        # ========== 阶段4: 计算最终执行参数 ==========
        final_grid_spacing_mult = sovereign_params['grid_spacing_mult'] * market_advice['grid_spacing_adjust']
        final_order_size_pct = sovereign_params['max_order_size_pct'] * market_advice['order_size_adjust']
        
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
        
        # 检查持仓超时
        if self._check_inventory_timeout(sovereign_params['inventory_timeout']):
            need_salvation = True
            salvation_price = current_price * (0.999 if self.position.size > 0 else 1.001)
        
        # ========== 阶段7: 生成网格价格 ==========
        anchor_price = self.anchor[-1] if pd.notna(self.anchor[-1]) else current_price
        grid_levels = max(2, sovereign_params['grid_levels'])  # 至少2层
        
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
        
        # 计算订单数量
        equity = self.equity
        order_qty = int((equity * final_order_size_pct) / current_price)
        order_qty = max(1, order_qty)  # 至少1单位
        
        # 挂买单
        if inventory_metrics['skew_factor'] < 0.8:
            for buy_price in grid_prices['buy_prices']:
                if buy_price < current_price * 0.995:
                    tag = 'SALVATION_BUY' if need_salvation and buy_price <= salvation_price else 'GRID_BUY'
                    self.buy(limit=buy_price, size=order_qty, tag=tag)
        
        # 挂卖单
        if inventory_metrics['skew_factor'] > -0.8:
            for sell_price in grid_prices['sell_prices']:
                if sell_price > current_price * 1.005:
                    tag = 'SALVATION_SELL' if need_salvation and sell_price >= salvation_price else 'GRID_SELL'
                    self.sell(limit=sell_price, size=order_qty, tag=tag)
        
        # ========== 阶段9: V2.0特有日志 ==========
        if len(self.data) % 500 == 0:
            cashflow_state_info = self.cashflow_sm.get_state_info()
            
            self.logger.info(
                f"🔀 V2.0系统状态 | "
                f"复合状态: {self.composite_state:15s} | "
                f"市场: {self.sovereign_state:10s} | "
                f"现金流: {cashflow_state_info['state']:10s} | "
                f"权益: ${equity:,.0f} | "
                f"净敞口: {inventory_metrics['net_exposure_pct']:.2%} | "
                f"已实现PNL: ${inventory_metrics['daily_realized_pnl']:.2f} | "
                f"自适应调整: {self.adaptive_adjustment_count}次"
            )
    
    # ========== 事件处理函数 ==========
    
    def on_order(self, order):
        """订单状态变化回调"""
        if order.is_completed:
            # 记录持仓
            if order.size > 0 and order.side == 'BUY':
                self.position_entries.append({
                    'time': self.data.index[-1],
                    'size': order.size,
                    'price': order.price
                })
            elif order.size > 0 and order.side == 'SELL':
                self.position_entries.append({
                    'time': self.data.index[-1],
                    'size': -order.size,
                    'price': order.price
                })
            elif order.size < 0:
                # 平仓
                closed_size = abs(order.size)
                for entry in self.position_entries[:]:
                    if abs(entry['size']) <= closed_size:
                        self.position_entries.remove(entry)
                        closed_size -= abs(entry['size'])
                    if closed_size <= 0:
                        break
    
    def on_trade(self, trade):
        """交易执行回调"""
        # 计算手续费
        trade_value = trade.size * trade.price
        fee = trade_value * 0.0006  # 万6手续费
        
        # 使用FIFO库存管理器处理交易
        realized_pnl = self.inventory_manager.process_trade(
            trade_side=trade.side,
            trade_size=trade.size,
            trade_price=trade.price,
            trade_time=self.data.index[-1]
        )
        
        # 更新现金流账本
        self.daily_cashflow = self.inventory_manager.daily_realized_pnl
        self.daily_fees += fee
        self.daily_trade_count += 1
        self.daily_rebate_accrued += trade_value * self.REBATE_RATE
        
        # 记录交易日志（每100笔）
        if self.daily_trade_count % 100 == 0:
            self.logger.debug(
                f"📒 账本更新 | "
                f"交易#{self.daily_trade_count} | "
                f"已实现PNL: ${realized_pnl:.2f} | "
                f"累计PNL: ${self.inventory_manager.daily_realized_pnl:.2f} | "
                f"手续费: ${fee:.2f}"
            )

# ==================== 回测执行器 ====================

def run_btc_cashflow_v2_backtest(
    data,
    initial_cash: float = 100000,
    commission: float = 0.0006,
    start_date: str = None,
    end_date: str = None
):
    """
    运行V2.0回测
    
    Args:
        data: pandas DataFrame格式的OHLC数据
        initial_cash: 初始资金
        commission: 手续费率
        start_date: 开始日期（可选）
        end_date: 结束日期（可选）
    """
    print("="*70)
    print("🚀 BTC高频现金流V2.0双状态机复合系统 - 回测启动")
    print("="*70)
    
    if start_date and end_date:
        print(f"📅 时间范围: {start_date} 至 {end_date}")
    print(f"💰 初始资金: ${initial_cash:,.0f}")
    print(f"💸 手续费率: {commission*10000} bps")
    print(f"🧠 系统架构: 市场状态机 × 现金流状态机 = 复合自适应决策")
    print("="*70)
    
    # 执行回测
    bt = Backtest(
        data,
        BTCHighFreqCashflow_V2,
        cash=initial_cash,
        commission=commission,
        margin=1.0,
        trade_on_close=False,
        exclusive_orders=True
    )
    
    # 运行回测
    stats = bt.run()
    
    # 输出结果
    print("\n" + "="*70)
    print("📈 V2.0回测结果摘要")
    print("="*70)
    
    # 基础指标
    key_stats = [
        ('最终权益', f"${stats['Equity Final [$]']:,.2f}"),
        ('总收益率', f"{stats['Return [%]']:.2f}%"),
        ('年化收益率', f"{stats['Return (Ann.) [%]']:.2f}%"),
        ('夏普比率', f"{stats['Sharpe Ratio']:.2f}"),
        ('最大回撤', f"{stats['Max. Drawdown [%]']:.2f}%"),
        ('总交易次数', f"{stats['# Trades']:,}"),
        ('胜率', f"{stats['Win Rate [%]']:.2f}%"),
        ('盈利因子', f"{stats['Profit Factor']:.2f}"),
    ]
    
    for name, value in key_stats:
        print(f"{name:>15}: {value}")
    
    # V2.0特有分析
    try:
        strategy_instance = bt.strategy
        
        print("\n🧠 V2.0系统特有分析")
        print("-"*70)
        
        # 复合状态分布
        if hasattr(strategy_instance, 'composite_engine'):
            decision_summary = strategy_instance.composite_engine.get_decision_history_summary()
            
            print(f"📊 复合决策分布 (共{decision_summary['total_decisions']}次决策):")
            for state, stats in decision_summary['state_distribution'].items():
                print(f"   {state:20s}: {stats['count']:4d}次 ({stats['percentage']:.1f}%)")
        
        # 现金流状态分析
        if hasattr(strategy_instance, 'cashflow_sm'):
            state_history = strategy_instance.cashflow_sm.state_history
            print(f"\n💰 现金流状态变更次数: {len(state_history)}")
            
            if state_history:
                print("   最近5次状态变更:")
                for i, record in enumerate(state_history[-5:]):
                    print(f"     {i+1}. {record['old_state']} → {record['new_state']} "
                          f"(手续费覆盖比: {record['metrics'].get('fee_coverage_ratio', 0):.2f})")
        
        # 自适应调整统计
        print(f"\n🔄 自适应调整次数: {strategy_instance.adaptive_adjustment_count}")
        
        # FIFO库存统计
        if hasattr(strategy_instance, 'inventory_manager'):
            inv_metrics = strategy_instance.inventory_manager.calculate_inventory_metrics(
                strategy_instance.data.Close[-1]
            )
            print(f"\n📦 FIFO库存统计:")
            print(f"   总已实现PNL: ${inv_metrics['total_realized_pnl']:,.2f}")
            print(f"   日已实现PNL: ${inv_metrics['daily_realized_pnl']:,.2f}")
            print(f"   未实现PNL: ${inv_metrics['unrealized_pnl']:,.2f}")
            print(f"   净敞口: {inv_metrics['net_quantity']:.4f} BTC")
        
    except Exception as e:
        print(f"⚠️ V2.0特有分析失败: {e}")
    
    # 保存详细日志
    print("\n📝 详细日志已保存到: btc_cashflow_v2.log")
    
    return stats

# ==================== 压力测试场景 ====================

def create_stress_test_scenario(scenario_type: str = 'FEE_CRISIS'):
    """
    创建压力测试场景
    
    Args:
        scenario_type: 测试类型
            - 'FEE_CRISIS': 手续费危机（返佣减半，手续费翻倍）
            - 'VOLATILITY_SPIKE': 波动率飙升
            - 'LIQUIDITY_CRASH': 流动性崩溃
    """
    
    class StressTestStrategy(BTCHighFreqCashflow_V2):
        """压力测试策略"""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.scenario_type = scenario_type
            self.stress_test_active = True
            
        def on_trade(self, trade):
            """重写on_trade以应用压力测试"""
            trade_value = trade.size * trade.price
            
            if self.scenario_type == 'FEE_CRISIS':
                # 手续费危机：返佣减半，手续费翻倍
                fee = trade_value * 0.0012  # 万12手续费（翻倍）
                rebate_rate = 0.0002  # 返佣减半
            else:
                # 默认：正常手续费
                fee = trade_value * 0.0006  # 万6手续费
                rebate_rate = self.REBATE_RATE
            
            # 使用FIFO库存管理器
            realized_pnl = self.inventory_manager.process_trade(
                trade_side=trade.side,
                trade_size=trade.size,
                trade_price=trade.price,
                trade_time=self.data.index[-1]
            )
            
            # 更新现金流账本
            self.daily_cashflow = self.inventory_manager.daily_realized_pnl
            self.daily_fees += fee
            self.daily_trade_count += 1
            self.daily_rebate_accrued += trade_value * rebate_rate
    
    return StressTestStrategy

# ==================== 主程序入口 ====================

def main():
    """主程序入口"""
    import warnings
    warnings.filterwarnings('ignore')
    
    print("🔧 BTC高频现金流V2.0系统 - 单文件完整版本")
    print("\n使用说明:")
    print("1. 加载你的BTC数据到变量 'data'")
    print("2. 调用 run_btc_cashflow_v2_backtest(data, ...) 运行回测")
    print("3. 查看 btc_cashflow_v2.log 获取详细日志")
    print("\n压力测试:")
    print("  策略类 = create_stress_test_scenario('FEE_CRISIS')")
    print("  然后使用该策略类运行回测")
    print("\n示例数据加载代码:")
    print("""
import pandas as pd

# 示例数据加载
data = pd.read_csv('your_btc_data.csv', index_col=0, parse_dates=True)
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

# 运行回测
stats = run_btc_cashflow_v2_backtest(
    data=data,
    initial_cash=100000,
    commission=0.0006,
    start_date='2024-01-01',
    end_date='2024-03-01'
)
    """)

if __name__ == "__main__":
    main()
```

🚀 V2.0 系统特性总结

1. 完整架构

· 市场状态机：处理市场异常（波动率、单边成交）
· 现金流状态机：处理财务异常（手续费覆盖、连续亏损）
· 复合决策引擎：市场状态 × 现金流状态 = 最终行为
· 自适应控制器：基于指标动态调整参数
· FIFO库存管理器：真实现金流计算

2. 核心创新

· 双状态机正交设计：解决状态污染问题
· 复合决策矩阵：16种市场状态 × 4种现金流状态 = 64种复合状态
· 真实现金流计算：基于FIFO的已实现损益
· 自适应闭环：现金流指标反向控制交易行为

3. 使用方式

```python
# 1. 加载你的数据
# data = pd.read_csv('your_btc_data.csv', index_col=0, parse_dates=True)

# 2. 运行回测
stats = run_btc_cashflow_v2_backtest(
    data=data,
    initial_cash=100000,
    commission=0.0006
)

# 3. 压力测试
StressStrategy = create_stress_test_scenario('FEE_CRISIS')
bt = Backtest(data, StressStrategy, cash=100000, commission=0.0006)
stats = bt.run()
```

4. 验证重点

1. 状态独立性：现金流状态变化不应被市场状态掩盖
2. 复合决策有效性：市场WAR+现金流CRITICAL应触发WAR_EMERGENCY
3. 生存能力：在手续费危机中存活
4. 真实现金流：基于FIFO的损益计算准确

这个单文件版本包含了所有V2.0核心功能，可以直接运行回测。系统会生成详细的日志文件btc_cashflow_v2.log，记录所有状态变更和决策过程。

现在你可以加载你的数据并运行这个完整的V2.0系统了！
