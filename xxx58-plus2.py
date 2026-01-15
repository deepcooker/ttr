import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import logging
import sys
import os

from .backtest_utils import BacktestUtils
from .quant import QuantitativeDiagnosis

# ==================== Survival Grid V3.0: Asymmetric Defense ====================
class SurvivalGridV3(Strategy):
    """
    Pool A V3.0: 非对称防御网格
    
    【核心升级】
    1. 非对称网格：暴跌时，Bid网格拉得比Ask更宽，防止接飞刀。
    2. 波动率放大器：ATR 越高，Bid 间距放大倍数越高。
    """
    
    # --- 1. 资金管理 ---
    base_order_pct = 0.02      
    max_net_exposure = 0.20    
    
    # --- 2. 网格结构 ---
    anchor_period = 20         
    grid_spacing_atr_mult = 0.8 
    min_grid_spacing_pct = 0.008 
    
    # --- 3. 风险引擎 ---
    war_atr_threshold = 0.03   
    stop_trading_atr_threshold = 0.05 
    war_spacing_mult = 3.0     
    
    # --- 4. 逃生参数 ---
    salvation_skew_threshold = 0.5 
    min_salvation_profit = 0.0015 
    
    # --- 5. 🔥 V3 新增: 非对称防御参数 ---
    # 当 ATR 升高时，Bid 间距额外放大的系数
    # Bid_Step = Base_Step * (1 + ATR_Pct * crash_protection_factor)
    crash_protection_factor = 20.0 

    def init(self):
        self.logger = logging.getLogger("SurvivalGridV3")
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.anchor = self.I(talib.EMA, self.data.Close, self.anchor_period)
        self.war_mode_count = 0

    def get_avg_entry_price(self):
        if self.position.size == 0: return None
        total_cost = 0
        total_size = 0
        for trade in self.trades:
            if trade.size * self.position.size > 0:
                total_cost += abs(trade.size) * trade.entry_price
                total_size += abs(trade.size)
        if total_size == 0: return None
        return total_cost / total_size

    def on_bar(self):
        price = self.data.Close[-1]
        current_atr = self.atr[-1]
        current_anchor = self.anchor[-1]
        equity = self.equity
        
        if price <= 0: return

        atr_pct = current_atr / price
        
        # 净持仓
        net_position_size = self.position.size
        net_exposure_val = net_position_size * price
        net_exposure_pct = net_exposure_val / equity
        
        avg_entry_price = self.get_avg_entry_price()
        if avg_entry_price is None: avg_entry_price = price

        # ==================== Risk Engine ====================
        is_high_vol = atr_pct > self.war_atr_threshold
        is_high_exposure = abs(net_exposure_pct) > (self.max_net_exposure * 0.8)
        is_war_mode = is_high_vol or is_high_exposure
        
        if is_war_mode:
            self.war_mode_count += 1
            current_spacing_mult = self.war_spacing_mult
        else:
            current_spacing_mult = 1.0
            
        # 熔断
        is_extreme_vol = atr_pct > self.stop_trading_atr_threshold
        allow_new_entry = not is_extreme_vol
            
        # Inventory Skew
        skew_factor = net_exposure_pct / self.max_net_exposure
        skew_factor = max(min(skew_factor, 1.0), -1.0)
        
        # ==================== Quoting (非对称核心) ====================
        base_step = max(self.min_grid_spacing_pct, atr_pct * self.grid_spacing_atr_mult)
        
        # 🔥 V3 核心: 计算非对称系数
        # 波动率越大，Bid 间距拉得越宽 (怕跌)
        # Ask 间距保持相对正常 (急着卖)
        bid_asymmetry = 1.0 + (atr_pct * self.crash_protection_factor)
        ask_asymmetry = 1.0 # 卖方保持原样，甚至可以缩小以便快速成交
        
        final_bid_step = base_step * current_spacing_mult * bid_asymmetry
        final_ask_step = base_step * current_spacing_mult * ask_asymmetry
        
        # 计算价格
        bid_price = current_anchor * (1 - final_bid_step * (1 + skew_factor))
        ask_price = current_anchor * (1 + final_ask_step * (1 - skew_factor))
        
        # ==================== Cost Salvation ====================
        if skew_factor > self.salvation_skew_threshold:
            salvation_price = avg_entry_price * (1 + self.min_salvation_profit)
            if ask_price > salvation_price:
                ask_price = max(salvation_price, price * 1.001) 
                
        elif skew_factor < -self.salvation_skew_threshold:
            salvation_price = avg_entry_price * (1 - self.min_salvation_profit)
            if bid_price < salvation_price:
                bid_price = min(salvation_price, price * 0.999)

        # ==================== Execution ====================
        for order in self.orders:
            order.cancel()
            
        qty = int((equity * self.base_order_pct) / price)
        if qty == 0: qty = 1
        
        # Bid
        if (skew_factor < 0) or (allow_new_entry and skew_factor < 0.95):
            if price > bid_price:
                tag = 'ESC_Bid' if (skew_factor < -0.5) else 'GRID_Bid'
                self.buy(limit=bid_price, size=qty, tag=tag)
        
        # Ask
        if (skew_factor > 0) or (allow_new_entry and skew_factor > -0.95):
            if price < ask_price:
                tag = 'ESC_Ask' if (skew_factor > 0.5) else 'GRID_Ask'
                self.sell(limit=ask_price, size=qty, tag=tag)

    next = on_bar

# ==================== 启动器 ====================
def run_survival_grid_v3():
    symbol = 'BTCUSDT'
    start_date = '2019-07-10'
    end_date = '2025-12-31'
    data_dir = '/root/autodl-tmp/policy/data/processed_1m'
    
    cash = 10000000
    commission = 0.0006 
    
    data = BacktestUtils.prepare_data(symbol, start_date, end_date, data_dir)
    if data is None: return

    print(f"\n🛡️ [Pool A] 启动 Survival Grid V3.0 (Asymmetric Defense)...")
    print(f"   - 核心升级: 暴跌时 Bid 网格非对称拉宽，拒绝接飞刀")
    
    bt = Backtest(
        data, 
        SurvivalGridV3,
        cash=cash, 
        commission=commission,
        margin=1,
        hedging=False,
        trade_on_close=False 
    )
    
    stats = bt.run()
    
    print("\n📊 [V3.0 统计]")
    try:
        diag = QuantitativeDiagnosis(stats, data)
        diag.print_report()
        BacktestUtils.format_with_chinese(stats, cash, commission)
    except Exception as e:
        print(f"诊断失败: {e}")

if __name__ == "__main__":
    run_survival_grid_v3()
    
'''
 [1] 时序层级分析 (Time Series)
--------------------------------------------------------------------------------
1.1 年度汇总:
          总盈亏(U)  交易次数  平均收益率
Year                         
2019 -272,697.11  1746  -0.00
2020  -54,959.18  4263  -0.00
2021  -34,602.68  7397  -0.00
2022 -485,126.42  4067  -0.00
2023 -144,583.27  1381  -0.00
2024 -222,452.29  1912  -0.00
2025  -51,777.71   826  -0.00

1.2 月度盈亏矩阵 (全量):
Month        1         2         3         4        5        6        7        8         9        10       11       12     Total
Year                                                                                                                            
2019         -         -         -         -        -        -   96,647  -20,066   -74,655  -134,456  -47,090  -93,077  -272,697
2020   -11,071   -32,656  -114,644  -136,003  -15,861   57,766   -5,597  105,405    17,803    29,196   34,958   15,747   -54,959
2021    64,321    65,736   -92,582   -19,176  218,400  -28,602  -38,796  -18,122  -164,039  -115,390   14,805   78,843   -34,603
2022    43,186  -250,820   -76,196   -19,722   19,003   12,709  -12,359   -4,035  -147,262   -32,862   -5,641  -11,128  -485,126
2023    10,409   -31,302  -100,703   -59,319   36,783   13,386   16,270  -30,223    12,181   -45,645   63,968  -30,388  -144,583
2024   -52,189     5,864    67,588   -72,310  -57,868  -38,847   12,436  -23,529     1,272    10,153  -52,135  -22,886  -222,452
2025     4,493   -29,918   -30,098    39,808  -30,739    7,612   -5,273   14,936    -1,258   -27,083    7,033   -1,292   -51,778

1.3 日度盈亏日历 (最近10个交易日):
             当日盈亏(U)  次数    状态
Date                          
2025-12-30  3,452.32   2  ✅ 盈利
2025-12-29  3,618.47   2  ✅ 盈利
2025-12-26  6,270.37   6  ✅ 盈利
2025-12-24  1,539.60   1  ✅ 盈利
2025-12-22  2,487.87   2  ✅ 盈利
2025-12-19    574.13  11  ✅ 盈利
2025-12-18  6,966.47   7  ✅ 盈利
2025-12-17  2,015.86   3  ✅ 盈利
2025-12-16 -8,201.46   4  ❌ 亏损
2025-12-15 -8,930.58   2  ❌ 亏损

[2] 连胜连败深度透视 (Streaks & Psychology)
--------------------------------------------------------------------------------
🔥 连胜分析:
   - 最大连胜次数: 37 次
   - 连胜期间总赚: 50,251.81 U (平均每单 1,358.16 U)
   - 发生时间: 2019-07-21 ~ 2019-07-23

❄️ 连败分析 (最长):
   - 最大连败次数: 36 次
   - 连败期间总亏: -109,377.38 U (平均每单 -3,038.26 U)

❄️ 连败分析 (最痛 - 亏钱最多):
   - 连败次数: 20 次
   - 期间总亏: -258,319.18 U
   - 发生时间: 2020-03-12 ~ 2020-03-13

[3] 交易结构与持仓 (Structure)
--------------------------------------------------------------------------------
类型       次数     胜率%      平均盈亏         平均持仓       盈亏比     
Long     11194  57.9     70           8.2        0.80    
Short    10398  53.4     -197         8.4        0.68    

⏳ 持仓行为:
   - 盈利单平均持仓: 7.1h
   - 亏损单平均持仓: 9.9h

[4] 惨案现场取证 (Top Losses Forensics)
--------------------------------------------------------------------------------
Time             Dir   PnL(U)     MAE%   MFE%   | Indicators
--------------------------------------------------------------------------------
20-03-12 23:08   Long  -29201     -22.0  0.7    | 
20-03-12 10:30   Long  -27714     -22.9  0.5    | 
20-03-12 23:02   Long  -25766     -22.4  0.6    | 
20-03-12 23:18   Long  -25593     -20.2  0.6    | 
19-09-24 18:48   Long  -24397     -17.0  0.1    | 
20-03-12 22:29   Long  -23867     -23.3  1.2    | 
19-10-26 00:21   Short -20829     -18.5  0.6    | 
20-03-12 23:19   Long  -20790     -20.0  0.6    | 
19-10-26 00:23   Short -19784     -18.3  0.3    | 
20-03-13 01:49   Long  -19317     -16.3  0.5    | 
====================================================================================================
回测开始时间         : 2019-07-10 11:49:00+00:00
回测结束时间         : 2025-12-31 15:59:00+00:00
回测持续时长         : 2366 days 04:10:00
持仓时间占比(%)      : 95.53%
最终账户权益(USDT)   : 8733801.36
账户权益峰值(USDT)   : 10140788.02
总手续费(USDT)     : 2477564.70
总收益率(%)        : -12.66%
买入持有收益率(%)     : 577.13%
年化收益率(%)       : -2.03%
年化波动率(%)       : 3.24%
复合年增长率(%)      : -2.07%
夏普比率           : -0.63
索提诺比率          : -0.72
卡尔玛比率          : -0.14
阿尔法系数(%)       : -17.83%
贝塔系数           : 0.01
最大回撤(%)        : -14.37%
平均回撤(%)        : -0.08%
最大回撤持续时长       : 2335 days 05:59:00
平均回撤持续时长       : 9 days 00:44:00
总交易次数          : 21592.00
胜率(%)          : 55.70%
最佳单交易收益率(%)    : 21.35%
最差单交易收益率(%)    : -17.19%
平均单交易收益率(%)    : -0.09%
最长交易时长         : 26 days 19:35:00
平均交易时长         : 0 days 08:18:00
盈利因子           : 0.92
预期收益率(%)       : -0.07%
系统质量数          : -3.69
凯利准则           : -0.04
使用策略           : SurvivalGridV3
权益曲线           :                                  Equity  DrawdownPct   DrawdownDuration
datetime                                                               
2019-07-10 11:49:00+00:00  1.000000e+07     0.000000                NaT
2019-07-10 11:50:00+00:00  1.000000e+07     0.000000                NaT
2019-07-10 12:04:00+00:00  1.000000e+07     0.000000                NaT
2019-07-10 12:05:00+00:00  1.000000e+07     0.000000                NaT
2019-07-10 12:06:00+00:00  1.000000e+07     0.000000                NaT
...                                 ...          ...                ...
2025-12-31 15:55:00+00:00  8.733801e+06     0.138745                NaT
2025-12-31 15:56:00+00:00  8.733801e+06     0.138745                NaT
2025-12-31 15:57:00+00:00  8.733801e+06     0.138745                NaT
2025-12-31 15:58:00+00:00  8.733801e+06     0.138745                NaT
2025-12-31 15:59:00+00:00  8.733801e+06     0.138745 2335 days 05:59:00

[3378133 rows x 3 columns]
交易记录           :        Size  EntryBar  ExitBar    EntryPrice     ExitPrice    SL    TP  ...                  ExitTime        Duration       Tag Entry_ATR(H,L,C,14) Exit_ATR(H,L,C,14) Entry_EMA(C,20) Exit_EMA(C,20)
0        15       125      209  12924.195197  12396.433098  None  None  ... 2019-07-10 15:33:00+00:00 0 days 01:24:00  GRID_Bid           24.440279          57.825560    13020.961103   12357.987146
1         1       130      209  12872.802420  12396.433098  None  None  ... 2019-07-10 15:33:00+00:00 0 days 01:19:00  GRID_Bid           29.340779          57.825560    12977.632247   12357.987146
2        14       130      210  12872.802420  12410.156966  None  None  ... 2019-07-10 15:34:00+00:00 0 days 01:20:00  GRID_Bid           29.340779          56.695163    12977.632247   12363.893132
3         2       134      210  12798.554532  12410.156966  None  None  ... 2019-07-10 15:34:00+00:00 0 days 01:16:00  GRID_Bid           39.687708          56.695163    12916.549216   12363.893132
4        13       134      211  12798.554532  12425.858590  None  None  ... 2019-07-10 15:35:00+00:00 0 days 01:17:00  GRID_Bid           39.687708          54.716937    12916.549216   12369.808072
...     ...       ...      ...           ...           ...   ...   ...  ...                       ...             ...       ...                 ...                ...             ...            ...
21587    -1   3370118  3370868  89183.846467  87463.986091  None  None  ... 2025-12-26 14:55:00+00:00 0 days 12:30:00  GRID_Ask          160.484783         167.438411    88405.626502   88085.788994
21588     1   3370869  3374419  87354.098680  89094.682809  None  None  ... 2025-12-29 02:06:00+00:00 2 days 11:10:00  GRID_Bid          165.085668          98.437936    88020.228137   88587.181402
21589     1   3370873  3374420  87097.882821  89187.409041  None  None  ... 2025-12-29 02:07:00+00:00 2 days 11:07:00  GRID_Bid          166.537030          98.913798    87789.977090   88641.240316
21590     1   3370874  3376625  86988.018064  88770.405036  None  None  ... 2025-12-30 14:52:00+00:00 3 days 23:51:00  GRID_Bid          166.505814         131.932824    87712.636415   88199.189159
21591     1   3370874  3376626  86988.018064  88868.915330  None  None  ... 2025-12-30 14:53:00+00:00 3 days 23:52:00  GRID_Bid          166.505814         136.794765    87712.636415   88259.856858

[21592 rows x 18 columns]
权益曲线时间范围: 2019-07-10 11:49:00+00:00 ~ 2025-12-31 15:59:00+00:00
初始权益: 10000000.00 USDT
最终权益: 8733801.36 USDT
权益峰值: 10140788.02 USDT
权益谷值: 0.00 USDT
((venv) ) (base) root@autodl-container-179442a4e1-d8b048f8:~/a9quant# 
'''