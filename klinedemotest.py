import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy

from .backtest_utils import BacktestUtils
from .quant import QuantitativeDiagnosis

# --- 策略类：修复方法名兼容问题，保留所有原有逻辑 ---
class SimpleTimeframeTestStrategy(Strategy):
    # 保留你的原时间框架参数（核心验证日线1D与1分钟数据的映射）
    kline_period = "1D"
    resample_rule = '1D'
    cycle_minutes = 1440  # 日线对应1440分钟（1天=24*60）

    # 【修复1】将 on_init() 重命名为 init()，适配backtesting.py官方版本
    def init(self):
        # 保留你的原周期映射表（完整复刻，用于时间框架校验）
        self.period_map = {
            '1m':  {'rule': '1T',  'minutes': 1},
            '5m':  {'rule': '5T',  'minutes': 5},
            '10m': {'rule': '10T', 'minutes': 10},
            '15m': {'rule': '15T', 'minutes': 15},
            '30m': {'rule': '30T', 'minutes': 30},
            '1h':  {'rule': '1h',  'minutes': 60},
            '4h':  {'rule': '4h',  'minutes': 240},
            '1d':  {'rule': '1D',  'minutes': 1440},
        }

        # 保留你的原参数清洗逻辑（核心：映射时间框架参数）
        input_period = str(getattr(self, 'kline_period', '1H')).lower()
        self.period_info = self.period_map.get(input_period, self.period_map['1d'])
        self.resample_rule = self.period_info['rule']
        self.cycle_minutes = self.period_info['minutes']

        # 【关键打印1】初始化阶段：时间框架参数确认
        print("="*60)
        print("📌 初始化 - 时间框架参数明细")
        print("="*60)
        print(f"目标大周期（kline_period）：{self.kline_period}")
        print(f"对应pandas resample规则：{self.resample_rule}")
        print(f"对应分钟数（1周期=多少分钟）：{self.cycle_minutes}")
        print(f"周期名称：{input_period.upper()}")

        # 初始化大周期指标（保留你的原逻辑）
        self._init_indicators()

    def _init_indicators(self):
        # 保留你的原指标计算逻辑（1分钟数据聚合大周期高低点）
        high_s = pd.Series(self.data.High, index=self.data.index)
        low_s = pd.Series(self.data.Low, index=self.data.index)
        close_s = pd.Series(self.data.Close, index=self.data.index)

        # 核心：shift(1) 避免未来数据，聚合上一个大周期高低点
        self.p_high = high_s.resample(self.resample_rule).max().shift(1)
        self.p_low = low_s.resample(self.resample_rule).min().shift(1)

        # 绑定到self.I，对齐1分钟数据索引（backtesting.py规范）
        self.prev_high = self.I(lambda: self.p_high.reindex(close_s.index).ffill(), name='BreakHigh')
        self.prev_low = self.I(lambda: self.p_low.reindex(close_s.index).ffill(), name='BreakLow')

        # 【关键打印2】指标初始化阶段：大周期聚合结果验证
        print("\n" + "="*60)
        print("📌 指标初始化 - 大周期数据聚合结果")
        print("="*60)
        print(f"1分钟数据总长度：{len(close_s)} 根")
        print(f"大周期（{self.kline_period}）高点聚合结果长度：{len(self.p_high)}")
        print(f"大周期（{self.kline_period}）高点聚合时间索引：")
        print(f"   起始：{self.p_high.index[0] if len(self.p_high) > 0 else '无数据'}")
        print(f"   结束：{self.p_high.index[-1] if len(self.p_high) > 0 else '无数据'}")
        print(f"对齐后1分钟数据的大周期高点非空值数量：{self.prev_high[np.isfinite(self.prev_high)].size}")

    # 【修复2】将 on_next() 重命名为 next()，适配backtesting.py官方版本
    def next(self):
        try:
            # 保留你的原核心校验逻辑（数据量+无效数据过滤）
            if len(self.data) < self.cycle_minutes + 1:
                # 【可选打印】数据量不足时的日志（便于跟踪初始化过程）
                if len(self.data) % 360 == 0:  # 每60分钟打印一次进度
                    print(f"⏳ 数据量不足：当前{len(self.data)}根1分钟K线，需至少{self.cycle_minutes+1}根")
                return

            # 提取最新大周期高低点（保留你的原逻辑）
            upper = float(self.prev_high[-1])
            lower = float(self.prev_low[-1])

            # 无效数据过滤（保留你的原逻辑）
            if np.isnan(upper) or np.isnan(lower) or upper <= 0 or lower <= 0:
                if len(self.data) % 720 == 0:  # 每12小时打印一次，避免刷屏
                    print(f"⚠️  无效数据：upper={upper:.2f}，lower={lower:.2f}，跳过当前循环")
                return

            # 【关键打印3】有效数据阶段：时间框架映射验证（核心测试内容）
            current_time = self.data.index[-1]  # 当前1分钟K线的时间戳
            current_close = float(self.data.Close[-1])
            # 每360根K线打印一次（每6小时），避免日志刷屏
            if len(self.data) % 360 == 0:
                print("\n" + "-"*60)
                print(f"📌 有效数据验证 - 时间：{current_time}")
                print(f"- 当前1分钟K线收盘价：{current_close:.2f}")
                print(f"- 上一个{self.kline_period}高点（upper）：{upper:.2f}")
                print(f"- 上一个{self.kline_period}低点（lower）：{lower:.2f}")
                print(f"- 大周期高低点差值：{upper - lower:.2f}")
                print(f"- 当前1分钟数据累计：{len(self.data)}根（已满足大周期要求）")

        except Exception as e:
            print(f"❌ 运行异常：{e}")

# --- 回测运行入口：保留你的原调用格式，无自定义BacktestUtils ---
def run_cashflow_grid_v30():
    # 保留你的原参数（2天数据：2019-07-10 至 2019-07-11）
    symbol = 'BTCUSDT'
    start_date = '2019-07-10'
    end_date = '2019-07-13'  
    data_dir = '/root/autodl-tmp/policy/data/processed_1m'
    kline_period = '1D'
    cash = 1000000
    commission = 0.0006

    # 【关键打印0】回测启动参数
    print("🏁 启动时间框架测试回测（仅验证1分钟数据与大周期映射）")
    print("="*60)
    print(f"回测参数明细：")
    print(f"   - 标的：{symbol}")
    print(f"   - 时间范围：{start_date} ~ {end_date}（2天）")
    print(f"   - 数据目录：{data_dir}")
    print(f"   - 大周期信号：{kline_period}")
    print(f"   - 初始资金：{cash:,}")
    print(f"   - 手续费：{commission}")

    # 保留你的原数据加载调用（直接使用，依赖你已存在的BacktestUtils）
    print("\n" + "="*60)
    print("📌 开始加载1分钟数据（调用BacktestUtils.prepare_data）")
    print("="*60)
    data = BacktestUtils.prepare_data(
        symbol, 
        start_date, 
        end_date, 
        data_dir,
        target_timeframe='1m'  # 固定1分钟数据，贴合你的需求
    )

    # 数据校验
    if data is None:
        print("❌ 数据加载失败，退出回测")
        return
    print(f"\n✅ 数据加载成功：")
    print(f"   - 数据时间范围：{data.index[0]} ~ {data.index[-1]}")
    print(f"   - 1分钟K线总数：{len(data)} 根")
    print(f"   - 数据列：{list(data.columns)}")

    # 初始化回测（极简配置，仅用于验证时间框架）
    bt = Backtest(
        data, 
        SimpleTimeframeTestStrategy, 
        cash=cash, 
        commission=commission,
        margin=1.0,      
        hedging=True,    
        trade_on_close=False
    )

    # 运行回测（传入大周期参数）
    print("\n" + "="*60)
    print("📌 启动回测，执行时间框架验证")
    print("="*60)
    stats = bt.run(kline_period=kline_period)
    
    try:
        diag = QuantitativeDiagnosis(stats, data)
        diag.print_report()
        BacktestUtils.format_with_chinese(stats, cash, commission)
           
        #bt.plot(filename='/root/a9quant/strategies/backtest_result.html') 
           
    except Exception as e:
        print(f"诊断失败: {e}")
    
     

    # 【关键打印4】回测结束：核心统计（验证时间框架相关）
    print("\n" + "="*60)
    print("📊 回测结束 - 时间框架验证汇总")
    print("="*60)
    print(f"大周期（{kline_period}）对应分钟数：{SimpleTimeframeTestStrategy.cycle_minutes}")
    print(f"1分钟数据是否满足大周期要求：{'是' if len(data) >= SimpleTimeframeTestStrategy.cycle_minutes else '否'}")
    print(f"回测总运行步数（1分钟K线数）：{len(data)}")
    print(f"回测是否正常完成：{'是' if stats is not None else '否'}")

if __name__ == "__main__":
    run_cashflow_grid_v30()
    
    
'''
============================================================
📊 回测结束 - 时间框架验证汇总
============================================================
大周期（1D）对应分钟数：1440
1分钟数据是否满足大周期要求：是
回测总运行步数（1分钟K线数）：2156
回测是否正常完成：是
((venv) ) (base) root@autodl-container-179442a4e1-d8b048f8:~/a9quant# python -m strategies.hhhhh
🏁 启动时间框架测试回测（仅验证1分钟数据与大周期映射）
============================================================
回测参数明细：
   - 标的：BTCUSDT
   - 时间范围：2019-07-10 ~ 2019-07-13（2天）
   - 数据目录：/root/autodl-tmp/policy/data/processed_1m
   - 大周期信号：1D
   - 初始资金：1,000,000
   - 手续费：0.0006

============================================================
📌 开始加载1分钟数据（调用BacktestUtils.prepare_data）
============================================================

📂 加载数据 BTCUSDT 2019-07-10 -> 2019-07-13
   🎯 目标周期: 1m
   📦 数据源路径: /root/autodl-tmp/policy/data/processed_1m
📂 [DataFeed] 磁盘读取(未命中缓存): BTCUSDT | 20190710 -> 20190713 | 目标周期: 1m
✅ 数据加载完成：
   - K线周期: 1m
   - 数据行数: 5000
   - 时间范围: 2019-07-10 11:49:00+00:00 ~ 2019-07-13 23:59:00+00:00
   - 首条 High: 13075.5
   - 末条 High: 11378.0

✅ 数据加载成功：
   - 数据时间范围：2019-07-10 11:49:00+00:00 ~ 2019-07-13 23:59:00+00:00
   - 1分钟K线总数：5000 根
   - 数据列：['Open', 'High', 'Low', 'Close', 'Volume']

============================================================
📌 启动回测，执行时间框架验证
============================================================
============================================================
📌 初始化 - 时间框架参数明细
============================================================
目标大周期（kline_period）：1D
对应pandas resample规则：1D
对应分钟数（1周期=多少分钟）：1440
周期名称：1D

============================================================
📌 指标初始化 - 大周期数据聚合结果
============================================================
1分钟数据总长度：5000 根
大周期（1D）高点聚合结果长度：4
大周期（1D）高点聚合时间索引：
   起始：2019-07-10 00:00:00+00:00
   结束：2019-07-13 00:00:00+00:00
对齐后1分钟数据的大周期高点非空值数量：4284
Backtest.run:   0%|                                                                                                                                                                                            | 0/4283 [00:00<?, ?bar/s]⏳ 数据量不足：当前720根1分钟K线，需至少1441根
⏳ 数据量不足：当前1080根1分钟K线，需至少1441根
⏳ 数据量不足：当前1440根1分钟K线，需至少1441根

------------------------------------------------------------
📌 有效数据验证 - 时间：2019-07-11 18:03:00+00:00
- 当前1分钟K线收盘价：11622.50
- 上一个1D高点（upper）：13127.50
- 上一个1D低点（lower）：11569.00
- 大周期高低点差值：1558.50
- 当前1分钟数据累计：1800根（已满足大周期要求）

------------------------------------------------------------
📌 有效数据验证 - 时间：2019-07-12 00:03:00+00:00
- 当前1分钟K线收盘价：11253.00
- 上一个1D高点（upper）：12110.00
- 上一个1D低点（lower）：10999.50
- 大周期高低点差值：1110.50
- 当前1分钟数据累计：2160根（已满足大周期要求）

------------------------------------------------------------
📌 有效数据验证 - 时间：2019-07-12 06:03:00+00:00
- 当前1分钟K线收盘价：11397.00
- 上一个1D高点（upper）：12110.00
- 上一个1D低点（lower）：10999.50
- 大周期高低点差值：1110.50
- 当前1分钟数据累计：2520根（已满足大周期要求）

------------------------------------------------------------
📌 有效数据验证 - 时间：2019-07-12 12:03:00+00:00
- 当前1分钟K线收盘价：11652.00
- 上一个1D高点（upper）：12110.00
- 上一个1D低点（lower）：10999.50
- 大周期高低点差值：1110.50
- 当前1分钟数据累计：2880根（已满足大周期要求）

------------------------------------------------------------
📌 有效数据验证 - 时间：2019-07-12 18:39:00+00:00
- 当前1分钟K线收盘价：11570.00
- 上一个1D高点（upper）：12110.00
- 上一个1D低点（lower）：10999.50
- 大周期高低点差值：1110.50
- 当前1分钟数据累计：3240根（已满足大周期要求）

------------------------------------------------------------
📌 有效数据验证 - 时间：2019-07-13 00:39:00+00:00
- 当前1分钟K线收盘价：11765.00
- 上一个1D高点（upper）：11878.00
- 上一个1D低点（lower）：11079.00
- 大周期高低点差值：799.00
- 当前1分钟数据累计：3600根（已满足大周期要求）

------------------------------------------------------------
📌 有效数据验证 - 时间：2019-07-13 06:39:00+00:00
- 当前1分钟K线收盘价：11590.00
- 上一个1D高点（upper）：11878.00
- 上一个1D低点（lower）：11079.00
- 大周期高低点差值：799.00
- 当前1分钟数据累计：3960根（已满足大周期要求）

------------------------------------------------------------
📌 有效数据验证 - 时间：2019-07-13 12:39:00+00:00
- 当前1分钟K线收盘价：11360.00
- 上一个1D高点（upper）：11878.00
- 上一个1D低点（lower）：11079.00
- 大周期高低点差值：799.00
- 当前1分钟数据累计：4320根（已满足大周期要求）

------------------------------------------------------------
📌 有效数据验证 - 时间：2019-07-13 18:39:00+00:00
- 当前1分钟K线收盘价：11153.50
- 上一个1D高点（upper）：11878.00
- 上一个1D低点（lower）：11079.00
- 大周期高低点差值：799.00
- 当前1分钟数据累计：4680根（已满足大周期要求）
❌ 无交易记录，无法分析。                                                                                                                                                                                                                 
⚠️ 无交易记录
回测开始时间         : 2019-07-10 11:49:00+00:00
回测结束时间         : 2019-07-13 23:59:00+00:00
回测持续时长         : 3 days 12:10:00
持仓时间占比(%)      : 0.00%
最终账户权益(USDT)   : 1000000.00
账户权益峰值(USDT)   : 1000000.00
总收益率(%)        : 0.00%
买入持有收益率(%)     : -6.01%
年化收益率(%)       : 0.00%
年化波动率(%)       : 0.00%
复合年增长率(%)      : 0.00%
夏普比率           : N/A
索提诺比率          : N/A
卡尔玛比率          : N/A
阿尔法系数(%)       : 0.00%
贝塔系数           : 0.00
最大回撤(%)        : -0.00%
平均回撤(%)        : N/A%
最大回撤持续时长       : N/A
平均回撤持续时长       : N/A
总交易次数          : 0.00
胜率(%)          : N/A%
最佳单交易收益率(%)    : N/A%
最差单交易收益率(%)    : N/A%
平均单交易收益率(%)    : N/A%
最长交易时长         : N/A
平均交易时长         : N/A
盈利因子           : N/A
预期收益率(%)       : N/A%
系统质量数          : N/A
凯利准则           : N/A
使用策略           : SimpleTimeframeTestStrategy(kline_period=1D)
权益曲线           :                               Equity  DrawdownPct  DrawdownDuration
datetime                                                           
2019-07-10 11:49:00+00:00  1000000.0          0.0               NaN
2019-07-10 11:50:00+00:00  1000000.0          0.0               NaN
2019-07-10 12:04:00+00:00  1000000.0          0.0               NaN
2019-07-10 12:05:00+00:00  1000000.0          0.0               NaN
2019-07-10 12:06:00+00:00  1000000.0          0.0               NaN
...                              ...          ...               ...
2019-07-13 23:55:00+00:00  1000000.0          0.0               NaN
2019-07-13 23:56:00+00:00  1000000.0          0.0               NaN
2019-07-13 23:57:00+00:00  1000000.0          0.0               NaN
2019-07-13 23:58:00+00:00  1000000.0          0.0               NaN
2019-07-13 23:59:00+00:00  1000000.0          0.0               NaN

[5000 rows x 3 columns]
交易记录           : Empty DataFrame
Columns: [Size, EntryBar, ExitBar, EntryPrice, ExitPrice, SL, TP, PnL, Commission, ReturnPct, EntryTime, ExitTime, Duration, Tag]
Index: []
权益曲线时间范围: 2019-07-10 11:49:00+00:00 ~ 2019-07-13 23:59:00+00:00
初始权益: 1000000.00 USDT
最终权益: 1000000.00 USDT
权益峰值: 1000000.00 USDT
权益谷值: 0.00 USDT

============================================================
📊 回测结束 - 时间框架验证汇总
============================================================
大周期（1D）对应分钟数：1440
1分钟数据是否满足大周期要求：是
回测总运行步数（1分钟K线数）：5000
回测是否正常完成：是
((venv) ) (base) root@autodl-container-179442a4e1-d8b048f8:~/a9quant# 
'''