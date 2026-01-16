# htf_engine.py
"""
HTF Engine 时间语义说明 (Time Semantics)：

1. 核心机制：所有 HTF K 线均使用 resample + shift(1) + reindex(ffill)。
2. 业务含义：始终使用「上一根已完成的 HTF K 线」数据，绝不使用当前正在生成的 K 线。
3. 冷启动状态：
   - 在第一个 HTF 周期完成前（例如 4h 策略的前 4 小时），HTF OHLC / ATR / ADX 均为 NaN。
   - 策略层必须显式等待（使用 self.min_ready_bars 或 check NaN）。
4. 结论：这是严格“无未来函数”模型的必然行为，保证回测与实盘逻辑完全一致。
"""
import pandas as pd
import numpy as np


class HTFIndicatorEngine:
    """
    HTF 指标引擎（严禁未来函数）

    核心规则：
    1️⃣ 所有 HTF K 线 = resample + shift(1)
    2️⃣ on next 是 1m，但 HTF 永远取「已完成的上一根」
    """

    def __init__(self, bt_data, period_map: dict):
        # backtesting.py 的 data 不是 DataFrame，这里强转
        self.df = pd.DataFrame({
            'Open':  bt_data.Open,
            'High':  bt_data.High,
            'Low':   bt_data.Low,
            'Close': bt_data.Close,
        }, index=bt_data.index)

        self.period_map = period_map
        self.ohlc_cache = {}
        self.store = {}

    # =========================================================
    # HTF OHLC（永远是上一根完成 K）
    # =========================================================
    def get_ohlc(self, period: str) -> pd.DataFrame:
        period = period.lower()
        if period in self.ohlc_cache:
            return self.ohlc_cache[period]

        rule = self.period_map[period]['rule']

        ohlc = (
            self.df
            .resample(rule)
            .agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            })
            .shift(1)   # ⭐ 核心：只用“已完成的前一根”
        )

        self.ohlc_cache[period] = ohlc
        return ohlc

    # =========================================================
    # 前高 / 前低（HTF）
    # =========================================================
    def register_prev_high_low(self, period: str):
        ohlc = self.get_ohlc(period)

        high = ohlc['High'].reindex(self.df.index).ffill()
        low  = ohlc['Low'].reindex(self.df.index).ffill()

        self._register(period, 'prev_high', high)
        self._register(period, 'prev_low', low)

    # =========================================================
    # ATR（Wilder）
    # =========================================================
    def register_atr(self, period: str, window: int = 14):
        ohlc = self.get_ohlc(period)

        h, l, c = ohlc['High'], ohlc['Low'], ohlc['Close']
        pc = c.shift(1)

        tr = pd.concat([
            h - l,
            (h - pc).abs(),
            (l - pc).abs()
        ], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1 / window, adjust=False).mean()
        atr_1m = atr.reindex(self.df.index).ffill()

        self._register(period, 'atr', atr_1m)

    # =========================================================
    # ADX（Wilder）
    # =========================================================
    def register_adx(self, period: str, window: int = 14):
        ohlc = self.get_ohlc(period)

        h, l, c = ohlc['High'], ohlc['Low'], ohlc['Close']
        ph, pl, pc = h.shift(1), l.shift(1), c.shift(1)

        plus_dm = (h - ph).where((h - ph) > (pl - l), 0.0)
        minus_dm = (pl - l).where((pl - l) > (h - ph), 0.0)

        tr = pd.concat([
            h - l,
            (h - pc).abs(),
            (l - pc).abs()
        ], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1 / window, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False).mean() / atr

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(alpha=1 / window, adjust=False).mean()

        adx_1m = adx.reindex(self.df.index).ffill()
        self._register(period, 'adx', adx_1m)

    # =========================================================
    # 统一存取
    # =========================================================
    def _register(self, period: str, name: str, series: pd.Series):
        self.store.setdefault(period, {})[name] = series

    def get(self, period: str, name: str, n: int = 1):
        """
        n = 1 → 前一根 HTF
        n = 5 → 前 5 根 HTF（倒序）
        """
        series = self.store.get(period, {}).get(name)
        if series is None:
            return None

        if n == 1:
            val = series.iloc[-1]
            return None if pd.isna(val) else float(val)

        vals = series.dropna().iloc[-n:].tolist()
        return vals
