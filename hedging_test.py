import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
import logging

from .backtest_utils import BacktestUtils
from .quant_old import QuantitativeDiagnosis

class ValidatorStrategy(Strategy):
    # 参数：本金 1000万，每单 1 BTC (Int)
    amount = 1 
    
    def init(self):
        self.rebalance_wait = False # 隔空换弹标志
        self.last_action_time = None

    def next(self):
        # 1. 验证库存计算 (Inventory Audit)
        longs = [t for t in self.trades if t.size > 0]
        shorts = [t for t in self.trades if t.size < 0]
        
        long_qty = sum(t.size for t in longs)
        short_qty = sum(abs(t.size) for t in shorts)
        net_qty = long_qty - short_qty
        
        # 每 100 根 K 线打印一次状态，证明活着
        if len(self.data) % 100 == 0:
            print(f"[{self.data.index[-1]}] 持仓审计 -> 多: {long_qty} | 空: {short_qty} | 净: {net_qty}")
            print(f"    当前挂单数: {len(self.orders)}")

        # 2. 隔空换弹逻辑 (Cancel Wait)
        if self.rebalance_wait:
            # 到了这一步，说明上一帧已经 Cancel 了，保证金应该释放了
            # 我们在这里疯狂挂单，测试 Margin 是否正常
            self._spam_orders()
            self.rebalance_wait = False
            return

        # 3. 触发重置 (每 20 根 K 线重置一次，模拟网格调整)
        if len(self.data) % 20 == 0:
            if len(self.orders) > 0:
                # 全撤
                for o in self.orders: o.cancel()
                # 标记：下一帧再挂单
                self.rebalance_wait = True
            else:
                # 没单子，直接挂
                self._spam_orders()

        # 4. 独立止盈 (验证 Trade Close)
        # 只要有单子浮盈 > 0.5%，就平掉
        price = self.data.Close[-1]
        for t in self.trades:
            if t.size > 0 and (price - t.entry_price) / t.entry_price > 0.005:
                t.close()
            elif t.size < 0 and (t.entry_price - price) / t.entry_price > 0.005:
                t.close()

    def _spam_orders(self):
        """挂一堆单子，测试保证金占用"""
        price = self.data.Close[-1]
        # 上下各挂 5 单
        for i in range(1, 6):
            # 卖单 (Int Size)
            self.sell(limit=int(price * (1 + 0.001 * i)), size=1)
            # 买单 (Int Size)
            self.buy(limit=int(price * (1 - 0.001 * i)), size=1)

if __name__ == "__main__":
   
    
    symbol = 'BTCUSDT'
    # 随便找一段数据
    start_date = '2024-01-01'
    end_date = '2024-01-03' 
    data_dir = '/root/autodl-tmp/policy/data/processed_1m'
    cash=10000000
    commission=0.0006
    
    print(f"🌊 加载数据: {start_date} ~ {end_date}")
    data = BacktestUtils.prepare_data(symbol, start_date, end_date, data_dir)
    
    if data is not None:
        print("🚀 启动 错题本验证脚本...")
        
        # 关键配置：1000万本金，Hedging，50倍杠杆
        bt = Backtest(
            data, 
            ValidatorStrategy, 
            cash=cash, 
            commission=commission, 
            hedging=True, 
            margin=1
        )
        
        stats = bt.run()
        print("\n✅ 验证结束！如果中间没有报错且有交易，说明路通了。")
        print(f"总交易次数: {stats['# Trades']}")
        
        BacktestUtils.format_with_chinese(stats, cash, commission)
    
        try:
            diag = QuantitativeDiagnosis(stats, data)
            diag.print_report()
        except Exception as e:
            print(f"诊断失败: {e}")
            
            
'''
测开始时间         : 2024-01-01 00:00:00+00:00
回测结束时间         : 2024-01-03 23:59:00+00:00
回测持续时长         : 2 days 23:59:00
持仓时间占比(%)      : 98.70%
最终账户权益(USDT)   : 10014629.85
账户权益峰值(USDT)   : 10019649.27
总手续费(USDT)     : 30048.94
总收益率(%)        : 0.15%
买入持有收益率(%)     : 1.24%
年化收益率(%)       : 95.47%
年化波动率(%)       : 31.65%
复合年增长率(%)      : 13.07%
夏普比率           : 3.02
索提诺比率          : 26.91
卡尔玛比率          : 45.92
阿尔法系数(%)       : 0.11%
贝塔系数           : 0.03
最大回撤(%)        : -2.08%
平均回撤(%)        : -0.11%
最大回撤持续时长       : 1 days 23:29:00
平均回撤持续时长       : 0 days 02:51:00
总交易次数          : 567.00
胜率(%)          : 100.00%
最佳单交易收益率(%)    : 2.61%
最差单交易收益率(%)    : 0.38%
平均单交易收益率(%)    : 0.56%
最长交易时长         : 2 days 08:03:00
平均交易时长         : 0 days 08:36:00
盈利因子           : N/A
预期收益率(%)       : 0.56%
系统质量数          : 38.87
凯利准则           : N/A
使用策略           : ValidatorStrategy
权益曲线           :                                  Equity  DrawdownPct DrawdownDuration
datetime                                                             
2024-01-01 00:00:00+00:00  1.000000e+07     0.000000              NaT
2024-01-01 00:01:00+00:00  1.000000e+07     0.000000              NaT
2024-01-01 00:02:00+00:00  1.000000e+07     0.000000              NaT
2024-01-01 00:03:00+00:00  1.000000e+07     0.000000              NaT
2024-01-01 00:04:00+00:00  1.000000e+07     0.000000              NaT
...                                 ...          ...              ...
2024-01-03 23:55:00+00:00  1.001357e+07     0.000607              NaT
2024-01-03 23:56:00+00:00  1.001263e+07     0.000700              NaT
2024-01-03 23:57:00+00:00  1.001311e+07     0.000653              NaT
2024-01-03 23:58:00+00:00  1.001408e+07     0.000555              NaT
2024-01-03 23:59:00+00:00  1.001463e+07     0.000501  0 days 07:56:00

[4320 rows x 3 columns]
交易记录           :      Size  EntryBar  ExitBar  EntryPrice  ExitPrice    SL    TP        PnL  Commission  ReturnPct                 EntryTime                  ExitTime        Duration   Tag
0       1        32       96     42396.0    42636.4  None  None  189.38056    51.01944   0.004467 2024-01-01 00:32:00+00:00 2024-01-01 01:36:00+00:00 0 days 01:04:00  None
1       1        29       97     42438.0    42699.2  None  None  210.11768    51.08232   0.004951 2024-01-01 00:29:00+00:00 2024-01-01 01:37:00+00:00 0 days 01:08:00  None
2      -1       101      170     42769.0    42533.0  None  None  184.81880    51.18120   0.004321 2024-01-01 01:41:00+00:00 2024-01-01 02:50:00+00:00 0 days 01:09:00  None
3      -1        97      196     42735.0    42514.0  None  None  169.85060    51.14940   0.003975 2024-01-01 01:37:00+00:00 2024-01-01 03:16:00+00:00 0 days 01:39:00  None
4      -1        96      204     42692.0    42460.0  None  None  180.90880    51.09120   0.004238 2024-01-01 01:36:00+00:00 2024-01-01 03:24:00+00:00 0 days 01:48:00  None
..    ...       ...      ...         ...        ...   ...   ...        ...         ...        ...                       ...                       ...             ...   ...
562    -1      4156     4267     42861.0    42645.0  None  None  164.69640    51.30360   0.003843 2024-01-03 21:16:00+00:00 2024-01-03 23:07:00+00:00 0 days 01:51:00  None
563    -1      4168     4270     42851.0    42618.0  None  None  181.71860    51.28140   0.004241 2024-01-03 21:28:00+00:00 2024-01-03 23:10:00+00:00 0 days 01:42:00  None
564    -1      4156     4271     42818.0    42603.5  None  None  163.24710    51.25290   0.003813 2024-01-03 21:16:00+00:00 2024-01-03 23:11:00+00:00 0 days 01:55:00  None
565     1      4270     4315     42599.0    42818.9  None  None  168.64926    51.25074   0.003959 2024-01-03 23:10:00+00:00 2024-01-03 23:55:00+00:00 0 days 00:45:00  None
566     1      4245     4315     42598.0    42818.9  None  None  169.64986    51.25014   0.003983 2024-01-03 22:45:00+00:00 2024-01-03 23:55:00+00:00 0 days 01:10:00  None

[567 rows x 14 columns]
权益曲线时间范围: 2024-01-01 00:00:00+00:00 ~ 2024-01-03 23:59:00+00:00
初始权益: 10000000.00 USDT
最终权益: 10014629.85 USDT
权益峰值: 10019649.27 USDT
权益谷值: 0.00 USDT

====================================================================================================
🔬 全维量化诊断报告 V5.0 (The God View)
====================================================================================================

[1] 时序层级分析 (Time Series)
--------------------------------------------------------------------------------
1.1 年度汇总:
         总盈亏(U)  交易次数  平均收益率
Year                        
2024 139,435.36   567   0.01

1.2 月度盈亏矩阵 (全量):
Month        1  2  3  4  5  6  7  8  9 10 11 12    Total
Year                                                    
2024   139,435  -  -  -  -  -  -  -  -  -  -  -  139,435

1.3 日度盈亏日历 (最近10个交易日):
             当日盈亏(U)   次数    状态
Date                           
2024-01-03 83,867.00  280  ✅ 盈利
2024-01-02 41,959.00  213  ✅ 盈利
2024-01-01 13,609.35   74  ✅ 盈利

[2] 连胜连败深度透视 (Streaks & Psychology)
--------------------------------------------------------------------------------
🔥 连胜分析:
   - 最大连胜次数: 567 次
   - 连胜期间总赚: 139,435.36 U (平均每单 245.92 U)
   - 发生时间: 2024-01-01 ~ 2024-01-03

[3] 交易结构与持仓 (Structure)
--------------------------------------------------------------------------------
类型       次数     胜率%      平均盈亏         平均持仓       盈亏比     
Long     249    100.0    199          2.7        0.00    
Short    318    100.0    283          13.2       0.00    

⏳ 持仓行为:
   - 盈利单平均持仓: 8.6h
   - 亏损单平均持仓: nanh

[4] 惨案现场取证 (Top Losses Forensics)
--------------------------------------------------------------------------------
🎉 无亏损记录
====================================================================================================
((venv) ) (base) root@autodl-container-08a344b9ef-6e58dffb:~/a9quant# 

'''