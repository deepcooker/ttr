# klinedemotest.py
import numpy as np
from backtesting import Backtest, Strategy

from .htf_engine import HTFIndicatorEngine
from .backtest_utils import BacktestUtils
from .quant import QuantitativeDiagnosis


class SimpleTimeframeTestStrategy(Strategy):

    kline_period = "1D"   # 主 HTF（可改）

    def init(self):
        self.period_map = {
            '15m': {'rule': '15T', 'minutes': 15},
            '1d':  {'rule': '1D',  'minutes': 1440},
        }

        self.used_htf = ['15m', '1d']

        self.htf = HTFIndicatorEngine(self.data, self.period_map)

        for p in self.used_htf:
            self.htf.register_adx(p)
        self.htf.register_atr('1d')
        self.htf.register_prev_high_low('1d')

        # ⭐ 最慢 HTF 决定启动时间
        self.min_ready_bars = max(
            self.period_map[p]['minutes']
            for p in self.used_htf
        ) + 1

        print("=" * 60)
        print("📌 策略初始化完成")
        print("   ▸ 执行周期：1分钟 on next")
        print(f"   ▸ 使用 HTF：{', '.join(self.used_htf)}")
        print(f"   ▸ 启动所需最少分钟数：{self.min_ready_bars}")

    def next(self):
        if len(self.data) < self.min_ready_bars:
            return

        adx_15m = self.htf.get('15m', 'adx')
        adx_1d  = self.htf.get('1d', 'adx')
        atr_1d  = self.htf.get('1d', 'atr')

        ph = self.htf.get('1d', 'prev_high')
        pl = self.htf.get('1d', 'prev_low')

        if None in (adx_15m, adx_1d, atr_1d, ph, pl):
            return

        if len(self.data) % 360 == 0:
            print("-" * 60)
            print(self.data.index[-1])
            print(f"ADX15m={adx_15m:.2f} | ADX1D={adx_1d:.2f}")
            print(f"ATR1D={atr_1d:.2f}")
            print(f"PrevHigh={ph:.2f} | PrevLow={pl:.2f}")


def main():
    symbol = 'BTCUSDT'
    start_date = '2019-07-10'
    end_date = '2019-07-13'
    data_dir = '/root/autodl-tmp/policy/data/processed_1m'

    data = BacktestUtils.prepare_data(
        symbol, start_date, end_date,
        data_dir, target_timeframe='1m'
    )

    bt = Backtest(
        data,
        SimpleTimeframeTestStrategy,
        cash=1_000_000,
        commission=0.0006,
        trade_on_close=False
    )

    stats = bt.run(kline_period='1D')

    try:
        QuantitativeDiagnosis(stats, data).print_report()
    except Exception as e:
        print("诊断失败:", e)


if __name__ == "__main__":
    main()
'''
((venv) ) (base) root@autodl-container-179442a4e1-d8b048f8:~/a9quant# python -m strategies.klinedemotest

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
============================================================
📌 策略初始化完成
   ▸ 执行周期：1分钟 on next
   ▸ 使用 HTF：15m, 1d
   ▸ 启动所需最少分钟数：1441
Backtest.run:   0%|                                                                                                                                                                                | 0/4999 [00:00<?, ?bar/s]------------------------------------------------------------
2019-07-11 18:03:00+00:00
ADX15m=38.34 | ADX1D=100.00
ATR1D=1474.54
PrevHigh=11878.00 | PrevLow=11079.00
------------------------------------------------------------
2019-07-12 00:03:00+00:00
ADX15m=38.34 | ADX1D=100.00
ATR1D=1474.54
PrevHigh=11878.00 | PrevLow=11079.00
------------------------------------------------------------
2019-07-12 06:03:00+00:00
ADX15m=38.34 | ADX1D=100.00
ATR1D=1474.54
PrevHigh=11878.00 | PrevLow=11079.00
------------------------------------------------------------
2019-07-12 12:03:00+00:00
ADX15m=38.34 | ADX1D=100.00
ATR1D=1474.54
PrevHigh=11878.00 | PrevLow=11079.00
------------------------------------------------------------
2019-07-12 18:39:00+00:00
ADX15m=38.34 | ADX1D=100.00
ATR1D=1474.54
PrevHigh=11878.00 | PrevLow=11079.00
------------------------------------------------------------
2019-07-13 00:39:00+00:00
ADX15m=38.34 | ADX1D=100.00
ATR1D=1474.54
PrevHigh=11878.00 | PrevLow=11079.00
------------------------------------------------------------
2019-07-13 06:39:00+00:00
ADX15m=38.34 | ADX1D=100.00
ATR1D=1474.54
PrevHigh=11878.00 | PrevLow=11079.00
------------------------------------------------------------
2019-07-13 12:39:00+00:00
ADX15m=38.34 | ADX1D=100.00
ATR1D=1474.54
PrevHigh=11878.00 | PrevLow=11079.00
------------------------------------------------------------
2019-07-13 18:39:00+00:00
ADX15m=38.34 | ADX1D=100.00
ATR1D=1474.54
PrevHigh=11878.00 | PrevLow=11079.00
'''