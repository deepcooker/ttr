import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

class QuantitativeDiagnosis:
    def __init__(self, stats, data):
        self.stats = stats
        self.trades = stats['_trades'].copy()
        self.equity_curve = stats['_equity_curve'].copy()
        self.data = data
        self.initial_capital = self.equity_curve.iloc[0]['Equity']
        
        # 预处理数据
        if not self.trades.empty:
            self.trades['EntryTime'] = pd.to_datetime(self.trades['EntryTime'])
            self.trades['ExitTime'] = pd.to_datetime(self.trades['ExitTime'])
            self.trades['Duration_Hours'] = (self.trades['ExitTime'] - self.trades['EntryTime']).dt.total_seconds() / 3600
            self.trades['Year'] = self.trades['ExitTime'].dt.year
            self.trades['Month'] = self.trades['ExitTime'].dt.month
            
            # 区分交易类型 (Tag解析)
            self.trades['Type'] = self.trades['Tag'].apply(lambda x: 'GAMBLE' if 'GAMBLE' in str(x) or 'LOTTERY' in str(x) or 'SVD' in str(x) else ('GRID' if 'GRID' in str(x) or 'SCALP' in str(x) else 'BASE'))

    def print_report(self):
        if self.trades.empty:
            print("❌ 无交易记录，无法分析。")
            return

        print("\n" + "="*60)
        print("🔬 顶级量化诊断报告 (Deep Diagnosis)")
        print("="*60)

        # 1. 盈亏归因分析 (Tag Breakdown)
        self._analyze_components()
        
        # 2. 时序稳定性分析 (Yearly/Monthly)
        self._analyze_time_stability()
        
        # 3. 连胜连败与心态压力 (Streaks)
        self._analyze_streaks()
        
        # 4. 持仓行为分析 (Holding Behavior)
        self._analyze_holding_behavior()
        
        # 5. 极端风险场景复盘 (Worst Case)
        self._analyze_worst_cases()
        
        print("="*60)

    def _analyze_components(self):
        print(f"\n🧩 [1] 策略成分归因 (你是靠什么赚钱的？)")
        print("-" * 50)
        
        # 按 Type 分组统计
        grouped = self.trades.groupby('Type').agg({
            'PnL': ['count', 'sum', 'mean'],
            'ReturnPct': ['mean', 'min', 'max'],
            'Duration_Hours': 'mean'
        })
        grouped.columns = ['次数', '总盈亏(U)', '单笔均盈', '平均收益率%', '最差收益%', '最佳收益%', '平均持仓(h)']
        
        # 计算胜率
        win_rates = self.trades.groupby('Type').apply(lambda x: (x['PnL'] > 0).sum() / len(x) * 100)
        grouped['胜率%'] = win_rates
        
        # 格式化输出
        print(grouped[['次数', '胜率%', '总盈亏(U)', '平均收益率%', '最差收益%', '平均持仓(h)']].to_string(float_format="{:.2f}".format))
        
        # 核心诊断
        total_pnl = self.trades['PnL'].sum()
        gamble_pnl = self.trades[self.trades['Type']=='GAMBLE']['PnL'].sum() if 'GAMBLE' in self.trades['Type'].values else 0
        grid_pnl = self.trades[self.trades['Type']=='GRID']['PnL'].sum() if 'GRID' in self.trades['Type'].values else 0
        
        print(f"\n👉 诊断结论:")
        if gamble_pnl < 0 and abs(gamble_pnl) > grid_pnl:
            print("⚠️ [警告] 赌博端(Gamble) 亏损过大，完全覆盖了网格端的利润。建议优化 SVD 开仓阈值或缩小赌注。")
        elif grid_pnl < 0:
            print("⚠️ [警告] 网格端(Grid) 正在失血，可能是手续费过高或逆势扛单导致。")
        else:
            print("✅ [健康] 结构良好。请关注两者比例是否协调。")

    def _analyze_time_stability(self):
        print(f"\n📅 [2] 时序稳定性 (你能在熊市活下来吗？)")
        print("-" * 50)
        
        # 按年统计
        yearly = self.trades.groupby('Year')['PnL'].sum()
        print("年度盈亏:")
        print(yearly.to_string(float_format="{:,.2f}".format))
        
        # 按月热力数据 (Pivot Table)
        monthly_pivot = self.trades.groupby(['Year', 'Month'])['PnL'].sum().unstack().fillna(0)
        
        # 统计亏损月份
        losing_months = (monthly_pivot < 0).sum().sum()
        total_months = monthly_pivot.count().sum()
        print(f"\n月度胜率: {(total_months - losing_months)/total_months*100:.1f}% ({total_months - losing_months}/{total_months} 月盈利)")
        
        # 找出亏损最惨的月份
        monthly_flat = self.trades.set_index('ExitTime').resample('M')['PnL'].sum()
        worst_month = monthly_flat.idxmin()
        worst_month_loss = monthly_flat.min()
        print(f"💀 最惨月份: {worst_month.strftime('%Y-%m')} | 亏损: {worst_month_loss:,.2f} U")

    def _analyze_streaks(self):
        print(f"\n🎢 [3] 连胜/连败压力测试 (心态会不会崩？)")
        print("-" * 50)
        
        pnl = self.trades['PnL'].values
        wins = pnl > 0
        
        # 计算连胜
        y = np.concatenate(([0], wins, [0]))
        idx = np.flatnonzero(y[1:] != y[:-1])
        streaks = (idx[1:] - idx[:-1])[::2] # 偶数位是连胜长度(如果第一个是胜)
        # 简单的迭代逻辑修复
        current_win_streak = 0
        max_win_streak = 0
        current_loss_streak = 0
        max_loss_streak = 0
        
        for p in pnl:
            if p > 0:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
                
        print(f"🔥 最大连胜次数: {max_win_streak}")
        print(f"❄️ 最大连败次数: {max_loss_streak}")
        
        if max_loss_streak > 10:
            print("⚠️ [警告] 连败超过10次！如果是马丁策略，这可能意味着爆仓风险；如果是趋势策略，意味着磨损严重。")

    def _analyze_holding_behavior(self):
        print(f"\n⏳ [4] 持仓行为分析 (是在做时间的朋友吗？)")
        print("-" * 50)
        
        avg_win_duration = self.trades[self.trades['PnL']>0]['Duration_Hours'].mean()
        avg_loss_duration = self.trades[self.trades['PnL']<=0]['Duration_Hours'].mean()
        
        print(f"盈利单平均持仓: {avg_win_duration:.1f} 小时")
        print(f"亏损单平均持仓: {avg_loss_duration:.1f} 小时")
        
        if avg_loss_duration > avg_win_duration * 10:
            print("⚠️ [严重警告] 亏损单持仓时间是盈利单的10倍以上！")
            print("   👉 典型特征：【死扛】。赚了就跑，亏了死拿。这是爆仓的前兆。")
        elif avg_win_duration > avg_loss_duration * 5:
            print("✅ [优秀] 盈利单持仓更久，说明拿得住趋势，且止损果断。")

    def _analyze_worst_cases(self):
        print(f"\n🚑 [5] 死亡现场回溯 (怎么亏的？)")
        print("-" * 50)
        
        worst_5 = self.trades.sort_values('PnL').head(5)
        for i, t in worst_5.iterrows():
            print(f"时间: {t['EntryTime']} -> {t['ExitTime']}")
            print(f"   标签: {t['Tag']} | 方向: {'多' if t['Size']>0 else '空'}")
            print(f"   价格: {t['EntryPrice']:.2f} -> {t['ExitPrice']:.2f} | 盈亏: {t['PnL']:.2f} U")
            
            # 简单的环境判断
            # 获取开仓时的 ATR 或 趋势 (需要 data 支持，这里简化判断)
            print("-" * 30)

