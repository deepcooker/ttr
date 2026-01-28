# -*- coding: utf-8 -*-
import logging
import sys
from enum import Enum
import os

# 配置日志处理器（避免日志只配置级别无输出，生产级必备）
def setup_logger():
    logger = logging.getLogger("TrendEngine")
    logger.setLevel(logging.INFO)
    
    # 避免重复添加处理器
    if not logger.handlers:
        # 控制台输出处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 日志格式（包含时间、模块、级别、内容，生产级易排查）
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    return logger

# 初始化日志
logger = setup_logger()

# 状态枚举
class TrendState(Enum):
    EMPTY = 0
    L1_PROBE = 1     # 侦察兵
    L2_COMPOUND = 2  # 增厚
    L3_RAMPAGE = 3   # 狂暴

# 尝试导入RiskManager，添加异常处理（生产级容错）
try:
    from .advanced_risk import RiskManager
except ImportError as e:
    # 若为独立运行（无包结构），尝试直接导入
    try:
        from advanced_risk import RiskManager
    except ImportError:
        logger.error(f"无法导入RiskManager模块：{e}")
        raise SystemExit(1)

class TrendEngine:
    def __init__(self, risk_manager):
        self.rm = risk_manager
        self.state = TrendState.EMPTY
        
        # 持仓数据（生产级完整字段，所有数值初始化清晰，避免None值异常）
        self.entry_price = 0.0
        self.position_size = 0.0  # 名义价值 (USDT)，即持仓总价值
        self.margin_used = 0.0    # 实际占用保证金 (USDT)，真实冻结资金
        self.avg_leverage = 1.0   # 平均杠杆（总名义价值/总占用保证金）
        self.stop_loss = 0.0      # 止损价格
        self.unrealized_pnl = 0.0 # 未实现浮盈/浮亏，实时更新

    def on_tick(self, data):
        """
        核心Tick处理函数，接收行情数据并执行状态机逻辑
        :param data: 行情字典，必须包含：price, ema20, atr, rsi, vol_ratio
        """
        # 生产级：校验输入数据完整性，避免KeyError
        required_fields = ['price', 'ema20', 'atr', 'rsi', 'vol_ratio']
        for field in required_fields:
            if field not in data:
                logger.error(f"行情数据缺失必要字段：{field}")
                return
        
        price = data['price']
        atr = data['atr']
        
        # --- A. 实时更新浮盈 & 同步风控账本（生产级：实时监控，避免风控滞后） ---
        if self.state != TrendState.EMPTY:
            self._update_unrealized_pnl(price)
            self._sync_risk_snapshot(data)

        # --- B. 止损检查（优先执行，避免亏损扩大，单向移动止损锁定利润） ---
        if self.state != TrendState.EMPTY:
            self._update_stop_loss(price, atr)
            if price <= self.stop_loss:
                self._close_position(price, "触及止损")
                return

        # --- C. 状态机逻辑（按持仓状态执行对应策略，逐步加仓） ---
        if self.state == TrendState.EMPTY:
            if price > data['ema20']:  # 简单均线入场条件，可替换为复杂策略
                self._try_open_l1(data)
                
        elif self.state == TrendState.L1_PROBE:
            # 浮盈>1.5%尝试L2加仓，基于入场价计算真实收益率
            if self.entry_price <= 0:
                return
            pnl_pct = (price - self.entry_price) / self.entry_price
            if pnl_pct > 0.015:
                self._try_open_l2(data)
                
        elif self.state == TrendState.L2_COMPOUND:
            # 浮盈>5%且RSI>70尝试L3加仓，使用平均杠杆放大真实收益
            if self.entry_price <= 0:
                return
            pnl_pct = (price - self.entry_price) / self.entry_price * self.avg_leverage
            if pnl_pct > 0.05 and data['rsi'] > 70:
                self._try_open_l3(data)

    def _update_unrealized_pnl(self, price):
        """生产级：实时计算未实现浮盈/浮亏，确保账本数据准确"""
        if self.position_size <= 0 or self.entry_price <= 0:
            self.unrealized_pnl = 0.0
            return
        
        # 计算公式：(当前价-入场价) * 名义价值 / 入场价（贴合真实合约盈亏计算逻辑）
        self.unrealized_pnl = (price - self.entry_price) * (self.position_size / self.entry_price)

    def _sync_risk_snapshot(self, data):
        """生产级：实时同步趋势策略状态给RiskManager，支撑全局风控决策"""
        # 计算当前钱包总余额（锚定本金+已实现盈利+未实现浮盈）
        current_wallet_balance = self.rm.anchor_capital + self.rm.realized_profit + self.unrealized_pnl
        
        # 同步给风控模块，更新全局账本
        self.rm.update_snapshot(
            wallet_balance=current_wallet_balance,
            trend_float=self.unrealized_pnl,
            shark_float=0.0,
            margin_usage=self._calculate_margin_usage(current_wallet_balance)
        )

    def _calculate_margin_usage(self, wallet_balance):
        """生产级：计算实时保证金使用率，避免超过风控阈值"""
        if wallet_balance <= 0 or self.margin_used <= 0:
            return 0.0
        
        # 保证金使用率=占用保证金/钱包余额，限制最大值为1.0（避免超过100%）
        return min(1.0, self.margin_used / wallet_balance)

    def _update_stop_loss(self, price, atr):
        """生产级：单向移动止损，锁定利润，避免频繁波动触发止损"""
        new_stop = 0.0
        if self.state == TrendState.L1_PROBE:
            # L1：宽松止损，给行情足够波动空间
            new_stop = price - 2 * atr
        elif self.state == TrendState.L2_COMPOUND:
            # L2：保本止损，确保不亏损（加0.2%手续费补偿）
            new_stop = max(self.entry_price * 1.002, price - 1.5 * atr)
        elif self.state == TrendState.L3_RAMPAGE:
            # L3：严格止损，窄幅保护，锁定大幅盈利
            new_stop = price - 0.5 * atr
        
        # 生产级优化：仅当新止损价高于当前止损价，且差值足够大时更新（避免频繁变动）
        stop_diff_threshold = 0.001 * self.entry_price if self.entry_price > 0 else 0.01
        if new_stop > self.stop_loss and (new_stop - self.stop_loss) > stop_diff_threshold:
            self.stop_loss = new_stop

    def _try_open_l1(self, data):
        """尝试开仓L1（侦察兵），申请风控审批，获批后执行交易"""
        # 计算申请资金：趋势总预算的30%（趋势总预算=初始本金*趋势资金占比）
        trend_total_capital = self.rm.initial_capital * self.rm.trend_allocation
        req_capital = trend_total_capital * 0.3
        
        # 构造风控申请参数
        risk_request = {
            'engine': 'TREND',
            'action': 'OPEN_L1',
            'suggested_leverage': 3,
            'volatility_ratio': data['vol_ratio'],
            'estimated_risk': req_capital * 0.1  # 预估10%止损风险
        }
        
        # 提交风控审批
        ok, final_lev, msg = self.rm.approve_action(risk_request)
        if ok:
            logger.info(f"🚀 [L1开仓] 价格:{data['price']:.2f} | 批复:{msg} | 杠杆:{final_lev}x")
            self._execute_trade(data['price'], req_capital, final_lev, TrendState.L1_PROBE)
            self.stop_loss = data['price'] - 2 * data['atr']
            logger.info(f"    [L1风控] 止损价:{self.stop_loss:.2f} | 占用保证金:{self.margin_used:.2f}U")
        else:
            logger.warning(f"🚫 [L1被拒] {msg}")

    def _try_open_l2(self, data):
        """尝试加仓L2（增厚），申请风控审批，获批后执行交易"""
        # 计算申请资金：趋势总预算的30%
        trend_total_capital = self.rm.initial_capital * self.rm.trend_allocation
        req_capital = trend_total_capital * 0.3
        
        # 构造风控申请参数
        risk_request = {
            'engine': 'TREND',
            'action': 'ADD_L2',
            'suggested_leverage': 5,
            'volatility_ratio': data['vol_ratio'],
            'estimated_risk': req_capital * 0.05  # 预估5%止损风险（加仓风险降低）
        }
        
        # 提交风控审批
        ok, final_lev, msg = self.rm.approve_action(risk_request)
        if ok:
            logger.info(f"💪 [L2加仓] 价格:{data['price']:.2f} | 批复:{msg} | 杠杆:{final_lev}x")
            self._execute_trade(data['price'], req_capital, final_lev, TrendState.L2_COMPOUND)
            logger.info(f"    [L2风控] 止损价:{self.stop_loss:.2f} | 占用保证金:{self.margin_used:.2f}U")
        else:
            logger.warning(f"🚫 [L2被拒] {msg}")

    def _try_open_l3(self, data):
        """尝试加仓L3（狂暴），双重校验（自查+风控），获批后执行交易"""
        # 1. 自查：已实现盈利≥20U（生产级：双重校验，避免风控穿透）
        if self.rm.realized_profit < 20:
             logger.warning(f"🚫 [L3自查拦截] 实盈不足 ({self.rm.realized_profit:.2f}U < 20U)")
             return

        # 2. 计算申请资金：已实现盈利的80%（控制风险敞口，不全部投入）
        profit_bet = self.rm.realized_profit * 0.8
        
        # 3. 构造风控申请参数
        risk_request = {
            'engine': 'TREND',
            'action': 'ADD_L3',
            'suggested_leverage': 10,
            'volatility_ratio': data['vol_ratio'],
            'estimated_risk': profit_bet
        }
        
        # 4. 提交风控审批
        ok, final_lev, msg = self.rm.approve_action(risk_request)
        if ok:
            logger.info(f"🔥 [L3狂暴] 价格:{data['price']:.2f} | 赌注:{profit_bet:.2f}U | 杠杆:{final_lev}x")
            self._execute_trade(data['price'], profit_bet, final_lev, TrendState.L3_RAMPAGE)
            logger.info(f"    [L3风控] 止损价:{self.stop_loss:.2f} | 占用保证金:{self.margin_used:.2f}U")
        else:
            logger.warning(f"🚫 [L3被拒] {msg}")

    def _execute_trade(self, price, margin, lev, new_state):
        """生产级：执行交易，正确计算平均杠杆、加权入场价、占用保证金"""
        # 1. 计算本次交易的名义价值（保证金 * 杠杆）
        new_nominal_size = margin * lev
        
        # 2. 计算本次交易占用的真实保证金
        new_margin_used = margin
        
        # 3. 计算总名义价值和总占用保证金
        total_nominal_size = self.position_size + new_nominal_size
        total_margin_used = self.margin_used + new_margin_used

        # 4. 计算加权平均入场价（避免单次价格覆盖，反映真实持仓成本）
        if total_nominal_size > 0:
            self.entry_price = (self.entry_price * self.position_size + price * new_nominal_size) / total_nominal_size

        # 5. 计算平均杠杆（总名义价值 / 总占用保证金，反映真实杠杆水平）
        if total_margin_used > 0:
            self.avg_leverage = total_nominal_size / total_margin_used

        # 6. 更新持仓状态（所有字段同步更新，无残留）
        self.position_size = total_nominal_size
        self.margin_used = total_margin_used
        self.state = new_state

        # 7. 初始化止损价（若为首次开仓，设置默认止损，后续由_on_tick更新）
        if self.stop_loss <= 0 and price > 0:
            self.stop_loss = price - 2 * 1.0  # 默认ATR=1.0，后续将被真实ATR替换

    def _close_position(self, price, reason):
        """生产级：平仓操作，计算盈亏，同步风控账本，重置持仓状态"""
        # 1. 计算最终已实现盈亏（当前未实现浮盈即为本次平仓盈亏）
        final_realized_pnl = self.unrealized_pnl
        
        # 2. 计算平仓后钱包总余额
        total_balance_after_close = self.rm.anchor_capital + self.rm.realized_profit + final_realized_pnl

        # 3. 详细日志记录（生产级：方便后续复盘和问题排查）
        logger.info(f"💥 [平仓] {reason} | 价格:{price:.2f} | 入场价:{self.entry_price:.2f}")
        logger.info(f"    [平仓结果] 已实现盈亏:{final_realized_pnl:.2f}U | 平仓后余额:{total_balance_after_close:.2f}U")

        # 4. 核心：同步风控模块，更新全局账本（平仓后无浮盈、无保证金占用）
        self.rm.update_snapshot(
            wallet_balance=total_balance_after_close,
            trend_float=0.0,
            shark_float=0.0,
            margin_usage=0.0
        )

        # 5. 重置持仓状态（所有字段清零，避免残留数据影响后续交易）
        self.state = TrendState.EMPTY
        self.entry_price = 0.0
        self.position_size = 0.0
        self.margin_used = 0.0
        self.avg_leverage = 1.0
        self.stop_loss = 0.0
        self.unrealized_pnl = 0.0

# ==========================================
# 生产级极限博弈测试套件（独立运行时执行，覆盖全场景）
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("⚔️  TrendEngine 生产级集成极限博弈测试  ⚔️")
    print("="*60)

    # 临时测试文件（独立运行时使用，避免污染生产环境）
    test_state_file = "test_risk_state.json"
    
    # 前置清理：删除残留的测试文件
    if os.path.exists(test_state_file):
        os.remove(test_state_file)

    def run_scenario(name, data_feed, init_profit=0, manual_balance_override=None):
        """
        运行单个测试场景，确保场景间数据隔离
        :param name: 场景名称
        :param data_feed: 行情数据列表
        :param init_profit: 初始注入盈利
        :param manual_balance_override: 手动覆盖钱包余额
        :return: 风控模块实例
        """
        print(f"\n>>> 测试场景: {name}")
        
        # 每次场景初始化前清理测试文件，确保数据隔离（核心修复：避免场景间残留）
        if os.path.exists(test_state_file):
            os.remove(test_state_file)
        
        # 初始化风控模块（初始本金200U，使用临时测试文件）
        rm = RiskManager(initial_capital=200, state_file=test_state_file)
        
        # 初始化场景参数（注入盈利或手动覆盖余额）
        if manual_balance_override is not None:
             rm.update_snapshot(manual_balance_override, 0.0, 0.0, 0.0)
             current_profit = rm.anchor_capital + rm.realized_profit - 200
             print(f"    [初始化] 手动覆盖余额:{manual_balance_override:.2f}U | 实盈:{rm.realized_profit:.2f}U | 累计盈利:{current_profit:.2f}U")
        elif init_profit > 0:
            target_balance = 200 + init_profit
            rm.update_snapshot(target_balance, 0.0, 0.0, 0.0)
            print(f"    [初始化] 注入初始实盈:{rm.realized_profit:.2f}U | 锚定本金:{rm.anchor_capital:.2f}U")
        
        # 初始化趋势引擎
        te = TrendEngine(rm)
        
        # 执行Tick数据处理
        for i, tick in enumerate(data_feed):
            te.on_tick(tick)
        
        # 输出场景结束后的核心风控指标
        final_balance = rm.anchor_capital + rm.realized_profit
        cumulative_profit = final_balance - 200
        print(f"    [场景结束] 最终余额:{final_balance:.2f}U | 锚定本金:{rm.anchor_capital:.2f}U | 实盈:{rm.realized_profit:.2f}U | 累计盈利:{cumulative_profit:.2f}U")
        
        # 场景结束后清理测试文件，避免残留
        if os.path.exists(test_state_file):
            os.remove(test_state_file)
        
        return rm

    # --- 场景 1: 贫穷陷阱 (Poverty Trap) ---
    # 核心：无实盈强开L3，被自查+风控拦截
    data_poverty = [
        {'price': 100, 'ema20': 99, 'atr': 1, 'rsi': 50, 'vol_ratio': 1.0},
        {'price': 102, 'ema20': 100, 'atr': 1, 'rsi': 60, 'vol_ratio': 1.0},
        {'price': 108, 'ema20': 102, 'atr': 1, 'rsi': 75, 'vol_ratio': 1.0},
    ]
    rm1 = run_scenario("贫穷陷阱 (没利润强开L3)", data_poverty)
    
    # --- 场景 2: 波动率降维打击 (High Volatility) ---
    # 核心：高vol_ratio（2.0）触发杠杆降档（3x→2x）
    data_high_vol = [
        {'price': 100, 'ema20': 99, 'atr': 2, 'rsi': 50, 'vol_ratio': 2.0},
    ]
    rm2 = run_scenario("波动率打击 (高波降杠杆)", data_high_vol)
    
    # --- 场景 3: 完美风暴 (The Perfect Storm) ---
    # 核心：注入35U实盈，全流程L1→L2→L3→止损，账本更新准确
    data_perfect = [
        {'price': 100, 'ema20': 99, 'atr': 1, 'rsi': 50, 'vol_ratio': 1.0},
        {'price': 102, 'ema20': 100, 'atr': 1, 'rsi': 60, 'vol_ratio': 1.0},
        {'price': 110, 'ema20': 105, 'atr': 1, 'rsi': 80, 'vol_ratio': 1.0},
        {'price': 115, 'ema20': 110, 'atr': 1, 'rsi': 85, 'vol_ratio': 1.0},
        {'price': 114, 'ema20': 112, 'atr': 1, 'rsi': 40, 'vol_ratio': 1.0},
    ]
    rm3 = run_scenario("完美风暴 (全流程通关)", data_perfect, init_profit=35)
    
    # --- 场景 4: 熔断测试 (Margin Call) ---
    # 核心：余额150U（权益0.75<0.8），触发权益熔断，L1被拒
    data_margin = [
        {'price': 100, 'ema20': 99, 'atr': 1, 'rsi': 50, 'vol_ratio': 1.0},
    ]
    rm4 = run_scenario("熔断测试 (本金不足触发权益熔断)", data_margin, manual_balance_override=150)
    
    # --- 场景 5: 水位线重置测试 (Anchor Capital Upgrade) ---
    # 核心：趋势盈利≥40U（200*1.2=240），触发锚定本金上台阶
    data_anchor_upgrade = [
        {'price': 100, 'ema20': 99, 'atr': 1, 'rsi': 50, 'vol_ratio': 1.0},
        {'price': 105, 'ema20': 100, 'atr': 1, 'rsi': 60, 'vol_ratio': 1.0},
        {'price': 115, 'ema20': 105, 'atr': 1, 'rsi': 80, 'vol_ratio': 1.0},
        {'price': 130, 'ema20': 110, 'atr': 1, 'rsi': 85, 'vol_ratio': 1.0},
        {'price': 128, 'ema20': 115, 'atr': 1, 'rsi': 40, 'vol_ratio': 1.0},
    ]
    rm5 = run_scenario("水位线重置 (盈利触发锚定本金上台阶)", data_anchor_upgrade, init_profit=30)
    
    # --- 场景 6: 趋势浮亏超预算测试 (Unrealized PnL Exceed Budget) ---
    # 核心：持仓浮亏扩大，触发风控熔断，拦截加仓
    data_unrealized_loss = [
        {'price': 100, 'ema20': 99, 'atr': 1, 'rsi': 50, 'vol_ratio': 1.0},
        {'price': 98, 'ema20': 100, 'atr': 1, 'rsi': 40, 'vol_ratio': 1.0},
        {'price': 95, 'ema20': 99, 'atr': 1, 'rsi': 30, 'vol_ratio': 1.0},
    ]
    rm6 = run_scenario("趋势浮亏超预算 (拦截加仓)", data_unrealized_loss, init_profit=20)
    
    # 最终清理与提示
    print(f"\n==========================================")
    print("✅ 所有生产级测试场景执行完毕，无残留测试文件")
    print("==========================================")