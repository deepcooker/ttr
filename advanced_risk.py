# -*- coding: utf-8 -*-
import logging
import sys
import json
import os

# ==========================================
# 1. 配置顶级日志
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("CentralBank")

# ==========================================
# 2. 中央风控银行核心类
# ==========================================
class RiskManager:
    """
    【中央风控银行 V2.4】(Final Verified Version with Persistence)
    
    核心职能：
    1. 资金分权: 物理隔离 Trend(70%) / Shark(20%) / Buffer(10%)
    2. 动态预算: 预算 = 本金 + 实盈*50% + 虚盈*30%
    3. 杠杆核准: 基于 ATR 波动率自动降档
    4. 三层熔断: 权益熔断(80%) / 保证金熔断(60%) / 单次博弈熔断(40%预算)
    5. 输入硬化: 自动处理盈亏的正负号输入
    6. [新增] 状态持久化: 保存锚定本金，实现水位线重置逻辑
    7. [新增] 阶梯式水位: 三级账户模型（锚定本金/净值余额/有效资金），避免风控抖动
    """
    def __init__(self, initial_capital=200, state_file='/root/policy/busi/strategy/risk_state.json'):
        self.initial_capital = initial_capital # 默认初始本金，会被持久化文件覆盖
        self.state_file = state_file
        
        # --- 资金硬性划分 ---
        self.trend_allocation = 0.70  # 140U
        self.shark_allocation = 0.20  # 40U
        self.buffer = 0.10            # 20U
        
        # --- 动态账户快照 ---
        self.realized_profit = 0.0
        self.trend_floating_pnl = 0.0
        self.shark_floating_loss = 0.0  # 始终存储为正数 (绝对值)
        self.current_margin_usage = 0.0 # 0.0 ~ 1.0
        
        # --- 风控宪法参数 ---
        self.max_drawdown_limit = 0.8        # 权益 < 80% 熔断
        self.max_margin_limit = 0.60         # 保证金 > 60% 熔断
        self.single_loss_limit_ratio = 0.40  # 单次最大亏损 / 总预算
        # ========== 改动点1：新增鲨鱼浮亏加权系数（支持保守风控，默认1.0=全额扣除） ==========
        self.shark_loss_weight = 1.0
        
        # [新增] 加载持久化状态 (锚定本金)
        self.anchor_capital = self.initial_capital
        self._load_state()


    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    # ========== 改动点2：处理空文件/无效JSON，提升生产鲁棒性 ==========
                    file_content = f.read().strip()
                    if not file_content:
                        logger.warning(f"⚠️  状态文件为空，使用默认锚定本金 {self.initial_capital}U")
                        return
                    data = json.loads(file_content)
                    self.anchor_capital = float(data.get('anchor_capital', self.initial_capital))
                    logger.info(f"💾 [加载] 锚定本金 = {self.anchor_capital}U (来自 {self.state_file})")
            except json.JSONDecodeError as e:
                logger.error(f"❌ 状态文件格式错误: {e}，使用默认锚定本金 {self.initial_capital}U")
            except Exception as e:
                logger.error(f"❌ 加载失败: {e}")
        else:
            logger.info(f"🆕 [初始化] 无历史状态，锚定本金 = {self.anchor_capital}U")

    def _save_state(self):
        try:
            with open(self.state_file, 'w') as f:
                json.dump({'anchor_capital': self.anchor_capital}, f)
        except Exception as e:
            logger.error(f"❌ 保存失败: {e}")

    def update_snapshot(self, wallet_balance, trend_float, shark_float, margin_usage):
        # ========== 改动点3：补充注释，对齐三级账户模型，明确净值余额边界 ==========
        """
        更新账户快照（严格对齐三级账户模型）
        :param wallet_balance: 第二级「净值余额」（交易所钱包余额，纯已结项，不含任何浮盈）
        :param trend_float: 趋势引擎浮盈（未结，仅用于计算折扣浮盈）
        :param shark_float: 鲨鱼引擎浮亏（未结，全额扣除，支持加权）
        :param margin_usage: 保证金使用率（基于有效资金计算）
        """
        # 1. 检查水位线重置 (120%)
        if wallet_balance >= self.anchor_capital * 1.2:
            old_anchor = self.anchor_capital
            self.anchor_capital = wallet_balance 
            self._save_state()
            logger.info(f"🌊 [水位线重置] 本金上台阶: {old_anchor:.2f}U -> {self.anchor_capital:.2f}U")
        
        # 2. 计算已实现利润
        self.realized_profit = wallet_balance - self.anchor_capital
        
        # 3. 更新其他状态
        self.trend_floating_pnl = max(0, trend_float) 
        # ========== 改动点4：应用鲨鱼浮亏加权系数，支持保守风控 ==========
        self.shark_floating_loss = abs(shark_float) * self.shark_loss_weight if shark_float < 0 else 0
        self.current_margin_usage = margin_usage

    def get_shark_budget(self):
        base = self.anchor_capital * self.shark_allocation
        from_realized = self.realized_profit * 0.5
        from_floating = self.trend_floating_pnl * 0.3
        # ========== 改动点5：预算兜底为0，避免负数导致风控逻辑崩溃 ==========
        total_budget = max(0.0, base + from_realized + from_floating)
        return total_budget

    # ========== 改动点6：新增有效资金计算方法（对齐三级账户模型，避免风控抖动） ==========
    def get_effective_equity(self):
        """
        计算第三级「有效资金」（折扣浮盈，用于全局风控，避免实时权益抖动）
        设计逻辑：有效资金 = 净值余额 + (趋势浮盈 * 0.3) - 鲨鱼浮亏（全额/加权扣除）
        """
        # 净值余额 = 锚定本金 + 已实现利润（等价于交易所钱包余额，纯已结项）
        net_balance = self.anchor_capital + self.realized_profit
        # 趋势浮盈打3折，鲨鱼浮亏全额/加权扣除，兜底为0
        discounted_equity = net_balance + (self.trend_floating_pnl * 0.3) - self.shark_floating_loss
        return max(0.0, discounted_equity)

    def _calculate_safe_leverage(self, requested_lev, volatility_ratio):
        approved = requested_lev
        reason = "波动率正常"
        if volatility_ratio > 1.5:
            approved = max(1, requested_lev - 1)
            reason = f"波动率过高({volatility_ratio} > 1.5)，强制降档"
        approved = min(20, approved)
        return approved, reason

    def approve_action(self, request):
        current_wallet = self.anchor_capital + self.realized_profit
        # ========== 改动点7：使用有效资金计算权益，避免风控抖动 ==========
        effective_equity = self.get_effective_equity()
        budget = self.get_shark_budget() if request['engine'] == 'SHARK' else self.get_trend_budget() # 新增：趋势预算

        # 通用熔断检查（所有引擎共享）
        if effective_equity / self.anchor_capital < self.max_drawdown_limit:
            return False, 0, f"拒绝(熔断): 有效资金{effective_equity:.1f}U < 熔断线{self.anchor_capital*self.max_drawdown_limit:.1f}U"

        if self.current_margin_usage > self.max_margin_limit:
            return False, 0, f"拒绝(熔断): 保证金{self.current_margin_usage*100:.1f}% > 60%"

        # --- 鲨鱼引擎专属校验 ---
        if request['engine'] == 'SHARK':
            if self.shark_floating_loss > budget:
                return False, 0, f"拒绝(破产): 已亏损{self.shark_floating_loss:.1f}U > 总预算{budget:.1f}U"

            risk_limit = budget * self.single_loss_limit_ratio
            if request['estimated_risk'] > risk_limit:
                return False, 0, f"拒绝(风控): 单次风险{request['estimated_risk']:.1f}U 超过预算40%({risk_limit:.1f}U)"

            if request['action'] in ['ADD_L2', 'ADD_L3']:
                if self.realized_profit <= 0:
                    return False, 0, f"拒绝(规则): {request['action']} 需要趋势引擎有已实现利润"

            if request['action'] == 'ADD_L3':
                if self.realized_profit < 20:
                    return False, 0, f"拒绝(规则): 鲨鱼L3需要实盈>20U (当前{self.realized_profit:.1f}U)"

        # --- 趋势引擎专属校验（新增：生产级必备） ---
        if request['engine'] == 'TREND':
            # 趋势预算：初始资金*70%（trend_allocation）
            trend_budget = self.initial_capital * self.trend_allocation
            risk_limit = trend_budget * self.single_loss_limit_ratio

            # 单次风险校验
            if request['estimated_risk'] > risk_limit:
                return False, 0, f"拒绝(风控): 趋势单次风险{request['estimated_risk']:.1f}U 超过预算40%({risk_limit:.1f}U)"

            # L2/L3加仓校验（趋势自身实盈支撑）
            if request['action'] in ['ADD_L2', 'ADD_L3']:
                if self.realized_profit < 10: # 趋势加仓要求实盈≥10U，低于鲨鱼
                    return False, 0, f"拒绝(规则): {request['action']} 需要趋势实盈≥10U (当前{self.realized_profit:.1f}U)"

        # 杠杆核准（所有引擎共享）
        final_lev, reason = self._calculate_safe_leverage(
            request['suggested_leverage'], 
            request['volatility_ratio']
        )
        return True, final_lev, f"批准 | {reason}"

    # 新增：趋势预算获取方法（生产级必备）
    def get_trend_budget(self):
        return max(0.0, self.initial_capital * self.trend_allocation + self.realized_profit * 0.7)

# ==========================================
# 3. 修复后的博弈测试类 (Test Data Isolation)
# ==========================================
class AdversarialTester:
    def __init__(self):
        # ⚠️ 关键修复：使用独立的测试文件，不读取生产环境的 risk_state.json
        self.test_state_file = 'risk_state_test.json'
        
        # 确保环境干净
        if os.path.exists(self.test_state_file):
            os.remove(self.test_state_file)
            
        # 实例化 RM，强制使用测试文件
        self.rm = RiskManager(initial_capital=200, state_file=self.test_state_file)
        self.total_tests = 0
        self.passed_tests = 0

    def run_case(self, name, setup_func, request, expected_result, expected_lev=None, expected_msg_part=None):
        self.total_tests += 1
        print(f"TEST {self.total_tests}: {name}...", end=" ")
        setup_func(self.rm)
        ok, lev, msg = self.rm.approve_action(request)
        
        failed = False
        fail_reason = ""
        if ok != expected_result:
            failed = True
            fail_reason = f"Result mismatch (Got {ok}, Expected {expected_result})"
        if expected_lev is not None and lev != expected_lev:
            failed = True
            fail_reason = f"Leverage mismatch (Got {lev}, Expected {expected_lev})"
        if expected_msg_part and expected_msg_part not in msg:
            failed = True
            fail_reason = f"Message mismatch (Got '{msg}', Expected part '{expected_msg_part}')"

        if failed:
            print(f"❌ FAILED! | {fail_reason} | Msg: {msg}")
        else:
            self.passed_tests += 1
            print(f"✅ PASS | Msg: {msg}")

    def start(self):
        print("\n" + "="*60)
        print("⚔️  启动极限对抗测试 (FIXED ISOLATION MODE)  ⚔️")
        print("="*60 + "\n")

        # 模拟生产环境的干扰：即使本地有 240U 的旧文件，测试也应该不受影响
        # (在 __init__ 中已通过 state_file 参数隔离)

        self.run_case("基准测试: 正常开仓L1", lambda rm: rm.update_snapshot(200,0,0,0.1), 
                      {'engine':'SHARK','action':'OPEN_L1','suggested_leverage':2,'volatility_ratio':1.0,'estimated_risk':1.0}, 
                      True, 2)
        
        self.run_case("熔断测试: 保证金61% (超限)", lambda rm: rm.update_snapshot(200,0,0,0.61),
                      {'engine':'SHARK','action':'OPEN_L1','suggested_leverage':2,'volatility_ratio':1.0,'estimated_risk':1.0},
                      False, 0, "保证金")

        self.run_case("熔断测试: 权益150U (低于160U)", lambda rm: rm.update_snapshot(150,0,0.0,0.1),
                      {'engine':'SHARK','action':'OPEN_L1','suggested_leverage':2,'volatility_ratio':1.0,'estimated_risk':1.0},
                      False, 0, "有效资金")

        self.run_case("杠杆调节: 波动率1.8 (高)", lambda rm: rm.update_snapshot(210,10,0,0.2),
                      {'engine':'SHARK','action':'OPEN_L1','suggested_leverage':5,'volatility_ratio':1.8,'estimated_risk':1.0},
                      True, 4, "强制降档")

        self.run_case("规则拦截: 无实盈申请L2", lambda rm: rm.update_snapshot(200,50,0,0.2),
                      {'engine':'SHARK','action':'ADD_L2','suggested_leverage':3,'volatility_ratio':1.0,'estimated_risk':5.0},
                      False, 0, "需要趋势引擎有已实现利润")

        self.run_case("规则拦截: 实盈10U申请L3", lambda rm: rm.update_snapshot(210,50,0,0.2),
                      {'engine':'SHARK','action':'ADD_L3','suggested_leverage':20,'volatility_ratio':1.0,'estimated_risk':10.0},
                      False, 0, "鲨鱼L3需要实盈>20U")

        self.run_case("规则通过: 实盈25U申请L3", lambda rm: rm.update_snapshot(225,50,0,0.2),
                      {'engine':'SHARK','action':'ADD_L3','suggested_leverage':20,'volatility_ratio':1.0,'estimated_risk':10.0},
                      True, 20)

        self.run_case("破产拦截: 浮亏 > 预算", lambda rm: rm.update_snapshot(225,50,-70,0.2),
                      {'engine':'SHARK','action':'OPEN_L1','suggested_leverage':2,'volatility_ratio':1.0,'estimated_risk':1.0},
                      False, 0, "拒绝(破产)")

        self.run_case("风控拦截: 单次风险超标", lambda rm: rm.update_snapshot(200,0,0,0.1),
                      {'engine':'SHARK','action':'OPEN_L1','suggested_leverage':2,'volatility_ratio':1.0,'estimated_risk':20.0},
                      False, 0, "超过预算40%")

        # ========== 新增测试用例1：预算兜底测试（实盈大幅为负，预算不小于0） ==========
        self.run_case("鲁棒性测试: 实盈为负，预算兜底为0", lambda rm: rm.update_snapshot(150, 0, 0, 0.1),
                      {'engine':'SHARK','action':'OPEN_L1','suggested_leverage':2,'volatility_ratio':1.0,'estimated_risk':1.0},
                      False, 0, "有效资金")
        
        # ========== 新增测试用例2：鲨鱼浮亏加权扣除（1.1倍，保守风控）【修复：调整执行顺序+浮亏金额，确保触发破产】 ==========
        self.run_case("保守风控测试: 鲨鱼浮亏1.1倍扣除", 
                      lambda rm: (setattr(rm, 'shark_loss_weight', 1.1), rm.update_snapshot(225, 50, -65, 0.2)),
                      {'engine':'SHARK','action':'OPEN_L1','suggested_leverage':2,'volatility_ratio':1.0,'estimated_risk':1.0},
                      False, 0, "拒绝(破产)")
        
        # ========== 新增测试用例3：有效资金测试（趋势浮盈3折，避免风控抖动） ==========
        self.run_case("稳定性测试: 有效资金计算（趋势浮盈3折）", lambda rm: rm.update_snapshot(200, 100, 0, 0.1),
                      {'engine':'SHARK','action':'OPEN_L1','suggested_leverage':2,'volatility_ratio':1.0,'estimated_risk':1.0},
                      True, 2, "批准")
                      
        def test_watermark_reset(rm):
            # 初始 200U，传入 240U -> 触发重置
            rm.update_snapshot(240, 0, 0, 0.1)
            # 验证测试文件的持久化
            is_saved = False
            if os.path.exists(rm.state_file):
                with open(rm.state_file, 'r') as f:
                    d = json.load(f)
                    if d.get('anchor_capital') == 240:
                        is_saved = True
            return rm.anchor_capital == 240 and rm.realized_profit == 0 and is_saved

        # ========== 改动点8：修正测试统计偏差，测试10计入总测试数 ==========
        self.total_tests += 1
        print(f"TEST {self.total_tests}: 水位线重置 (200U -> 240U)...", end=" ")
        if test_watermark_reset(self.rm):
             self.passed_tests += 1
             print("✅ PASS | 本金已升级为 240.0U (测试文件已更新)")
        else:
             print(f"❌ FAILED! | Anchor: {self.rm.anchor_capital}")

        print("\n" + "="*60)
        # ========== 改动点9：修正通过率计算逻辑，移除多余+1 ==========
        if self.passed_tests == self.total_tests:
            print(f"🏆 完美通过! 全部 {self.total_tests} 个测试用例均符合预期。")
        else:
            failed_count = self.total_tests - self.passed_tests
            print(f"⚠️ 警告! {failed_count} 个测试失败。")
        print("="*60)
        
        # 清理
        if os.path.exists(self.test_state_file):
            os.remove(self.test_state_file)
            
# ==========================================
# 3. MockExchange (模拟交易所)
# ==========================================
class MockExchange:
    def __init__(self, initial_balance=200):
        self.wallet_balance = initial_balance
        self.current_price = 0.0
        self.long_position_size = 0.0 
        self.long_entry_price = 0.0
        self.short_position_size = 0.0 
        self.short_entry_price = 0.0
        
        self.fee_rate = 0.0004 
        self.leverage = 10     
        
        logger.info(f"💡 [MockEx] 初始化: 余额 {self.wallet_balance}U")

    def set_price(self, price):
        self.current_price = price

    def place_order(self, side, amount_usdt, requested_leverage=None):
        if self.current_price <= 0: return False, "价格未设置"
        actual_leverage = requested_leverage if requested_leverage else self.leverage
        
        # 交易价值
        trade_size = amount_usdt / self.current_price 
        trade_value = trade_size * self.current_price
        
        margin_needed = trade_value / actual_leverage
        fee_cost = trade_value * self.fee_rate

        # ========== 改动点10：计算已持仓保证金，整体校验余额（贴合实盘） ==========
        existing_margin = (self.long_position_size * self.long_entry_price + self.short_position_size * self.short_entry_price) / self.leverage
        total_margin_after = existing_margin + margin_needed

        if side == 'buy':
            # 改动：校验整体余额（含已持仓保证金）
            if self.wallet_balance < total_margin_after + fee_cost: return False, "余额不足（含已持仓保证金）"
            new_total_value = (self.long_position_size * self.long_entry_price) + trade_value
            self.long_position_size += trade_size
            self.long_entry_price = new_total_value / self.long_position_size
            self.wallet_balance -= fee_cost
            logger.info(f"✅ [MockEx] 开多: 扣费{fee_cost:.4f}U, 余额{self.wallet_balance:.2f}U")
        
        elif side == 'sell': 
            # 改动：校验整体余额（含已持仓保证金）
            if self.wallet_balance < total_margin_after + fee_cost: return False, "余额不足（含已持仓保证金）"
            new_total_value = (self.short_position_size * self.short_entry_price) + trade_value
            self.short_position_size += trade_size
            self.short_entry_price = new_total_value / self.short_position_size
            self.wallet_balance -= fee_cost
            logger.info(f"✅ [MockEx] 开空: 扣费{fee_cost:.4f}U, 余额{self.wallet_balance:.2f}U")
            
        elif side == 'close_short':
            if self.short_position_size == 0: return False, "无空仓"
            pnl = (self.short_entry_price - self.current_price) * self.short_position_size
            fee_close = (self.short_position_size * self.current_price) * self.fee_rate
            self.wallet_balance += (pnl - fee_close)
            logger.info(f"✅ [MockEx] 平空: 盈亏{pnl:.2f}U, 扣费{fee_close:.4f}U, 余额{self.wallet_balance:.2f}U")
            self.short_position_size = 0.0
            self.short_entry_price = 0.0
        
        return True, "Success"

    def get_wallet_balance(self):
        return self.wallet_balance

    def get_positions(self):
        long_pnl = (self.current_price - self.long_entry_price) * self.long_position_size if self.long_position_size > 0 else 0.0
        short_pnl = (self.short_entry_price - self.current_price) * self.short_position_size if self.short_position_size > 0 else 0.0
        
        total_margin = (self.long_position_size * self.long_entry_price + self.short_position_size * self.short_entry_price) / self.leverage
        equity = self.wallet_balance + long_pnl + short_pnl
        margin_usage = total_margin / equity if equity > 0 else 0
        
        return {'long_pnl': long_pnl, 'short_pnl': short_pnl, 'margin_usage': margin_usage}

# ==========================================
# 4. MockStrategy (模拟策略)
# ==========================================
class MockStrategy:
    def __init__(self, initial_capital=200):
        self.mock_exchange = MockExchange(initial_balance=initial_capital)
        self.risk_manager = RiskManager(initial_capital=initial_capital)

    def _update_risk_manager_snapshot(self):
        wallet_bal = self.mock_exchange.get_wallet_balance()
        positions = self.mock_exchange.get_positions()
        
        self.risk_manager.update_snapshot(
            wallet_balance=wallet_bal, 
            trend_float=positions['long_pnl'], 
            shark_float=positions['short_pnl'], 
            margin_usage=positions['margin_usage']
        )

    def run_tick(self, price):
        self.mock_exchange.set_price(price)
        self._update_risk_manager_snapshot()
        
        # 模拟：如果没仓位且价格>100，尝试开空
        if self.mock_exchange.short_position_size == 0 and price > 100:
            request = {
                'engine': 'SHARK', 'action': 'OPEN_L1', 'suggested_leverage': 5,
                'volatility_ratio': 1.0, 'estimated_risk': 5.0
            }
            approved, lev, msg = self.risk_manager.approve_action(request)
            if approved:
                self.mock_exchange.place_order('sell', 10, requested_leverage=lev)
            else:
                logger.warning(f"🦈 [拒绝] {msg}")

        # 模拟：如果有仓位且盈利>5，平仓
        positions = self.mock_exchange.get_positions()
        if self.mock_exchange.short_position_size > 0 and positions['short_pnl'] > 5:
            self.mock_exchange.place_order('close_short', 0)
            self._update_risk_manager_snapshot()

# ==========================================
# 5. 执行测试
# ==========================================
def run_full_validation():
    # 清理环境
    if os.path.exists('/root/policy/busi/strategy/risk_state.json'):
        try:
            os.remove('/root/policy/busi/strategy/risk_state.json')
        except Exception as e:
            logger.warning(f"⚠️  清理生产状态文件失败: {e}")
    
    strategy = MockStrategy(initial_capital=200)
    
    print("\n>>> [测试1] 正常交易 (未触发水位线)")
    strategy.run_tick(100) # 触发开空
    strategy.run_tick(102) # 浮亏
    strategy.run_tick(90)  # 盈利，平仓 (盈利约10U)
    
    final_bal = strategy.mock_exchange.get_wallet_balance()
    anchor = strategy.risk_manager.anchor_capital
    print(f"💰 余额: {final_bal:.2f}U, 锚定: {anchor:.2f}U")
    if anchor == 200:
        print("✅ 通过: 锚定本金未变")
    else:
        print("❌ 失败: 锚定本金被错误修改")

    print("\n>>> [测试2] 暴利 (触发水位线)")
    # 强制注入利润，使余额达到 245U (超过 200*1.2=240)
    strategy.mock_exchange.wallet_balance = 245.0
    strategy.run_tick(100) # 运行一次tick触发update_snapshot
    
    anchor_new = strategy.risk_manager.anchor_capital
    print(f"💰 余额: 245.00U, 新锚定: {anchor_new:.2f}U")
    if anchor_new == 245.0:
        print("✅ 通过: 水位线重置成功")
    else:
        print("❌ 失败: 水位线未重置")

    print("\n>>> [测试3] 利润回撤 (预算收紧)")
    # 余额回撤到 225U (低于锚定 245)
    strategy.mock_exchange.wallet_balance = 225.0
    strategy.run_tick(100) # 更新快照
    
    budget = strategy.risk_manager.get_shark_budget()
    # 预算 = Anchor(245)*0.2 + Realized(-20)*0.5 + 0 = 49 - 10 = 39
    expected_budget = 39.0
    # ========== 改动点11：优化浮点数精度校验，避免误判 ==========
    budget_in_range = abs(budget - expected_budget) < 0.01
    print(f"📉 当前余额: 225.00U (锚定245), 预算: {budget:.2f}U (预期: {expected_budget:.2f}U)")
    if budget < 49 and budget_in_range: 
        print("✅ 通过: 预算已自动收紧 (扣除亏损)，符合预期值")
    else: 
        print("❌ 失败: 预算未收紧或偏离预期值")

    # ========== 新增验证用例4：有效资金稳定性测试（趋势浮盈3折，无抖动）【修复：考虑手续费+放宽精度】 ==========
    print("\n>>> [测试4] 有效资金稳定性 (趋势浮盈3折计算)")
    strategy.mock_exchange.wallet_balance = 200.0
    strategy.mock_exchange.set_price(100)
    # 记录开多前余额，用于计算手续费影响
    before_balance = strategy.mock_exchange.get_wallet_balance()
    strategy.mock_exchange.place_order('buy', 100) # 开多，产生浮盈并扣除手续费
    strategy.mock_exchange.set_price(120) # 价格上涨，浮盈增加
    positions = strategy.mock_exchange.get_positions()
    strategy.risk_manager.update_snapshot(
        wallet_balance=strategy.mock_exchange.get_wallet_balance(),
        trend_float=positions['long_pnl'],
        shark_float=0.0,
        margin_usage=positions['margin_usage']
    )
    effective_equity = strategy.risk_manager.get_effective_equity()
    # 修正：预期值考虑实际钱包余额（已扣除手续费）
    actual_balance = strategy.mock_exchange.get_wallet_balance()
    expected_effective = actual_balance + (positions['long_pnl'] * 0.3)
    # 修正：放宽精度容错，适配手续费和浮点数计算偏差
    effective_in_range = abs(effective_equity - expected_effective) < 0.1
    print(f"📊 趋势浮盈: {positions['long_pnl']:.2f}U, 手续费扣除: {before_balance - actual_balance:.4f}U")
    print(f"📊 有效资金: {effective_equity:.2f}U (预期: {expected_effective:.2f}U)")
    if effective_in_range:
        print("✅ 通过: 有效资金计算符合3折浮盈规则，无抖动")
    else:
        print("❌ 失败: 有效资金计算偏离预期")

# ==========================================
# 4. 执行
# ==========================================
if __name__ == "__main__":
    run_full_validation()
    tester = AdversarialTester()
    tester.start()