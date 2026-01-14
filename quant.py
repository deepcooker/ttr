import pandas as pd
import numpy as np
from datetime import timedelta
from itertools import groupby
import warnings

# 忽略 pandas 的一些切片警告
warnings.filterwarnings('ignore')

class QuantitativeDiagnosis:
    def __init__(self, stats, data=None):
        """
        全维量化诊断系统 V5.0 (The God View)
        :param stats: Backtesting.py 返回的 stats 对象
        :param data: 原始 OHLC 数据 (DataFrame)，必须包含 High, Low, Close，索引为时间
        """
        self.stats = stats
        # 兼容性处理：防止 stats['_trades'] 为空或索引问题
        self.trades = stats['_trades'].copy().reset_index(drop=True)
        self.equity_curve = stats['_equity_curve'].copy()
        self.data = data
        self.initial_capital = self.equity_curve.iloc[0]['Equity'] if not self.equity_curve.empty else 0
        
        # --- 核心预处理 ---
        if not self.trades.empty:
            self._preprocess_data()
            self._calculate_mae_mfe() # 核心增强：计算过程指标

    def _preprocess_data(self):
        """数据清洗与特征工程"""
        # 1. 强制时间类型转换 (去除时区以便计算)
        for col in ['EntryTime', 'ExitTime']:
            if col in self.trades.columns:
                # 兼容处理：如果是字符串先转datetime，如果是datetime直接去时区
                self.trades[col] = pd.to_datetime(self.trades[col])
                if self.trades[col].dt.tz is not None:
                    self.trades[col] = self.trades[col].dt.tz_localize(None)
        
        # 2. 重新计算精准时长 (小时)
        self.trades['Duration_Real'] = (self.trades['ExitTime'] - self.trades['EntryTime'])
        self.trades['Duration_Hours'] = self.trades['Duration_Real'].dt.total_seconds() / 3600.0
        
        # 3. 时间维度拆解
        self.trades['Year'] = self.trades['ExitTime'].dt.year
        self.trades['Month'] = self.trades['ExitTime'].dt.month
        self.trades['Date'] = self.trades['ExitTime'].dt.date
        self.trades['Hour'] = self.trades['EntryTime'].dt.hour
        
        # 4. 区分方向
        self.trades['Direction'] = np.where(self.trades['Size'] > 0, 'Long', 'Short')
        
        # 5. 策略标签解析 (Tag)
        if 'Tag' not in self.trades.columns:
            self.trades['Tag'] = 'BASE'
        
        def parse_tag(t):
            t_str = str(t).upper()
            if any(x in t_str for x in ['GAMBLE', 'LOTTERY', 'SVD']): return 'GAMBLE'
            if any(x in t_str for x in ['GRID', 'SCALP']): return 'GRID'
            return 'BASE'
        
        self.trades['Type'] = self.trades['Tag'].apply(parse_tag)
        
        # 6. 确保 ReturnPct 存在 (如果 Backtesting 没算)
        if 'ReturnPct' not in self.trades.columns:
            # 估算: PnL / (EntryPrice * Size)
            # 注意 Size 可能为负，取绝对值作为本金基数
            self.trades['ReturnPct'] = self.trades['PnL'] / (self.trades['EntryPrice'] * self.trades['Size'].abs())

    def _calculate_mae_mfe(self):
        """
        计算 MAE (最大不利变动) 和 MFE (最大有利变动)
        需要 self.data 支持
        """
        if self.data is None or self.trades.empty:
            self.trades['MAE_Pct'] = 0.0
            self.trades['MFE_Pct'] = 0.0
            return

        # 确保 data 索引无时区
        if self.data.index.tz is not None:
            self.data.index = self.data.index.tz_localize(None)

        mae_list = []
        mfe_list = []

        # 遍历计算 (虽然循环慢，但为了准确性是必须的)
        for _, row in self.trades.iterrows():
            try:
                # 切片获取持仓期间的行情
                mask = (self.data.index >= row['EntryTime']) & (self.data.index <= row['ExitTime'])
                df_period = self.data.loc[mask]
                
                if df_period.empty:
                    mae_list.append(0)
                    mfe_list.append(0)
                    continue
                
                entry_price = row['EntryPrice']
                
                if row['Direction'] == 'Long':
                    # 多单：最低价是最大亏损，最高价是最大浮盈
                    lowest = df_period['Low'].min()
                    highest = df_period['High'].max()
                    mae = (lowest - entry_price) / entry_price * 100
                    mfe = (highest - entry_price) / entry_price * 100
                else:
                    # 空单：最高价是最大亏损，最低价是最大浮盈
                    highest = df_period['High'].max()
                    lowest = df_period['Low'].min()
                    mae = (entry_price - highest) / entry_price * 100 # 负数代表亏损
                    mfe = (entry_price - lowest) / entry_price * 100
                
                mae_list.append(mae)
                mfe_list.append(mfe)
            except Exception:
                mae_list.append(0)
                mfe_list.append(0)
        
        self.trades['MAE_Pct'] = mae_list
        self.trades['MFE_Pct'] = mfe_list

    def print_report(self):
        if self.trades.empty:
            print("❌ 无交易记录，无法分析。")
            return

        print("\n" + "="*100)
        print("🔬 全维量化诊断报告 V5.0 (The God View)")
        print("="*100)

        # 1. 时序层级分析
        self._analyze_time_series()
        
        # 2. 连胜连败深度透视
        self._analyze_streaks_deep()

        # 3. 交易结构与持仓
        self._analyze_structure()
        
        # 4. 亏损深度透视
        self._analyze_forensics()
        
        print("="*100)

    def _analyze_time_series(self):
        print(f"\n[1] 时序层级分析 (Time Series)")
        print("-" * 80)
        
        # 1.1 Yearly
        yearly = self.trades.groupby('Year').agg({
            'PnL': ['sum', 'count'],
            'ReturnPct': 'mean'
        })
        yearly.columns = ['总盈亏(U)', '交易次数', '平均收益率']
        print("1.1 年度汇总:")
        print(yearly.to_string(float_format="{:,.2f}".format))
        
        # 1.2 Monthly Matrix (Full)
        print("\n1.2 月度盈亏矩阵 (全量):")
        monthly = self.trades.groupby(['Year', 'Month'])['PnL'].sum()
        # Reindex to ensure all months exist
        years = self.trades['Year'].unique()
        months = range(1, 13)
        idx = pd.MultiIndex.from_product([years, months], names=['Year', 'Month'])
        monthly = monthly.reindex(idx, fill_value=0).unstack()
        monthly['Total'] = monthly.sum(axis=1)
        
        # Format: 0 -> - (cleaner look), numbers -> formatted
        print(monthly.applymap(lambda x: '-' if x==0 else f"{x:,.0f}").to_string())

        # 1.3 Daily Calendar (Last 10 active days)
        print("\n1.3 日度盈亏日历 (最近10个交易日):")
        daily = self.trades.groupby('Date').agg({
            'PnL': 'sum',
            'Size': 'count' # count trades
        }).sort_index(ascending=False).head(10)
        daily.columns = ['当日盈亏(U)', '次数']
        daily['状态'] = daily['当日盈亏(U)'].apply(lambda x: '✅ 盈利' if x>0 else '❌ 亏损')
        print(daily.to_string(float_format="{:,.2f}".format))

    def _analyze_streaks_deep(self):
        print(f"\n[2] 连胜连败深度透视 (Streaks & Psychology)")
        print("-" * 80)
        
        # Identify streaks
        self.trades['Win'] = self.trades['PnL'] > 0
        # Group consecutive trades with same result
        streaks = []
        for is_win, group in groupby(self.trades.to_dict('records'), key=lambda x: x['PnL'] > 0):
            g_list = list(group)
            pnl_sum = sum(x['PnL'] for x in g_list)
            streaks.append({
                'Type': 'Win' if is_win else 'Loss',
                'Count': len(g_list),
                'Total_PnL': pnl_sum,
                'Avg_PnL': pnl_sum / len(g_list),
                'Start': g_list[0]['EntryTime'],
                'End': g_list[-1]['ExitTime']
            })
        
        df_streaks = pd.DataFrame(streaks)
        
        # Win Analysis
        wins = df_streaks[df_streaks['Type'] == 'Win']
        if not wins.empty:
            max_win = wins.loc[wins['Count'].idxmax()]
            print(f"🔥 连胜分析:")
            print(f"   - 最大连胜次数: {max_win['Count']} 次")
            print(f"   - 连胜期间总赚: {max_win['Total_PnL']:,.2f} U (平均每单 {max_win['Avg_PnL']:,.2f} U)")
            print(f"   - 发生时间: {max_win['Start'].strftime('%Y-%m-%d')} ~ {max_win['End'].strftime('%Y-%m-%d')}")
        
        # Loss Analysis
        losses = df_streaks[df_streaks['Type'] == 'Loss']
        if not losses.empty:
            max_loss = losses.loc[losses['Count'].idxmax()]
            # Find the streak with worst total PnL (might not be the longest)
            worst_pnl_streak = losses.loc[losses['Total_PnL'].idxmin()]
            
            print(f"\n❄️ 连败分析 (最长):")
            print(f"   - 最大连败次数: {max_loss['Count']} 次")
            print(f"   - 连败期间总亏: {max_loss['Total_PnL']:,.2f} U (平均每单 {max_loss['Avg_PnL']:,.2f} U)")
            
            if worst_pnl_streak['Count'] != max_loss['Count']:
                print(f"\n❄️ 连败分析 (最痛 - 亏钱最多):")
                print(f"   - 连败次数: {worst_pnl_streak['Count']} 次")
                print(f"   - 期间总亏: {worst_pnl_streak['Total_PnL']:,.2f} U")
                print(f"   - 发生时间: {worst_pnl_streak['Start'].strftime('%Y-%m-%d')} ~ {worst_pnl_streak['End'].strftime('%Y-%m-%d')}")

    def _analyze_structure(self):
        print(f"\n[3] 交易结构与持仓 (Structure)")
        print("-" * 80)
        
        # Long/Short Stats
        print(f"{'类型':<8} {'次数':<6} {'胜率%':<8} {'平均盈亏':<12} {'平均持仓':<10} {'盈亏比':<8}")
        for direction in ['Long', 'Short']:
            df = self.trades[self.trades['Direction'] == direction]
            if df.empty: continue
            
            count = len(df)
            win_rate = (df['PnL'] > 0).sum() / count * 100
            avg_pnl = df['PnL'].mean()
            avg_dur = df['Duration_Hours'].mean()
            
            wins = df[df['PnL']>0]['PnL'].mean() if not df[df['PnL']>0].empty else 0
            losses = abs(df[df['PnL']<=0]['PnL'].mean()) if not df[df['PnL']<=0].empty else 0
            rr = wins/losses if losses > 0 else 0
            
            print(f"{direction:<8} {count:<6} {win_rate:<8.1f} {avg_pnl:<12.0f} {avg_dur:<10.1f} {rr:<8.2f}")

        # Holding Behavior
        win_dur = self.trades[self.trades['PnL']>0]['Duration_Hours'].mean()
        loss_dur = self.trades[self.trades['PnL']<=0]['Duration_Hours'].mean()
        print(f"\n⏳ 持仓行为:")
        print(f"   - 盈利单平均持仓: {win_dur:.1f}h")
        print(f"   - 亏损单平均持仓: {loss_dur:.1f}h")
        if loss_dur > win_dur * 2:
            print("   ⚠️ 警告: 亏损单持仓时间显著长于盈利单 (死扛风险)")

    def _analyze_forensics(self):
        print(f"\n[4] 惨案现场取证 (Top Losses Forensics)")
        print("-" * 80)
        
        losses = self.trades[self.trades['PnL'] < 0].sort_values('PnL', ascending=True).head(10)
        if losses.empty:
            print("🎉 无亏损记录")
            return
            
        # Dynamic Indicators
        ind_cols = [c for c in self.trades.columns if any(x in c for x in ['SVD', 'Entropy', 'ADX', 'RSI', 'Strength'])]
        # Simplify names
        rename_map = {c: c.replace('Entry_', '').replace('_compute_', '')[:8] for c in ind_cols}
        
        print(f"{'Time':<16} {'Dir':<5} {'PnL(U)':<10} {'MAE%':<6} {'MFE%':<6} | {'Indicators'}")
        print("-" * 80)
        
        for _, row in losses.iterrows():
            t_str = row['EntryTime'].strftime('%y-%m-%d %H:%M')
            d_str = row['Direction']
            pnl_str = f"{row['PnL']:.0f}"
            mae_str = f"{row.get('MAE_Pct', 0):.1f}"
            mfe_str = f"{row.get('MFE_Pct', 0):.1f}"
            
            ind_str = " | ".join([f"{rename_map[c]}:{row[c]:.2f}" for c in ind_cols if pd.notnull(row[c])])
            
            print(f"{t_str:<16} {d_str:<5} {pnl_str:<10} {mae_str:<6} {mfe_str:<6} | {ind_str}")
            
            