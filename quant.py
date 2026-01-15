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
        全维量化诊断系统 V5.0 (Ultimate Optimized)
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
            self._calculate_mae_mfe_fast() # 🔥 使用极速版计算

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

    def _calculate_mae_mfe_fast(self):
        """
        🔥 极速版 MAE/MFE 计算 (Numpy Vectorized)
        性能提升 100x+
        """
        if self.data is None or self.trades.empty:
            self.trades['MAE_Pct'] = 0.0
            self.trades['MFE_Pct'] = 0.0
            return

        # 1. 准备数据 (Zero Copy if possible)
        # 确保 data 索引无时区且排序
        if self.data.index.tz is not None:
            self.data.index = self.data.index.tz_localize(None)
        if not self.data.index.is_monotonic_increasing:
            self.data = self.data.sort_index()

        times = self.data.index.values
        highs = self.data['High'].values
        lows = self.data['Low'].values
        
        entry_times = self.trades['EntryTime'].values
        exit_times = self.trades['ExitTime'].values
        entry_prices = self.trades['EntryPrice'].values
        directions = self.trades['Direction'].values
        
        # 2. 二分查找定位索引 (O(N log M))
        # side='left' 包含 Entry 当刻，side='right' 包含 Exit 当刻
        start_idxs = np.searchsorted(times, entry_times, side='left')
        end_idxs = np.searchsorted(times, exit_times, side='right')
        
        n = len(self.trades)
        mae_arr = np.zeros(n)
        mfe_arr = np.zeros(n)
        
        # 3. 循环切片计算 (O(N)) - Numpy Slice 极快
        for i in range(n):
            s, e = start_idxs[i], end_idxs[i]
            if s >= e: continue
            
            # 获取期间最高最低价
            p_high = highs[s:e].max()
            p_low = lows[s:e].min()
            entry = entry_prices[i]
            
            if directions[i] == 'Long':
                mae_arr[i] = (p_low - entry) / entry * 100
                mfe_arr[i] = (p_high - entry) / entry * 100
            else:
                mae_arr[i] = (entry - p_high) / entry * 100 # 空单：涨了是亏
                mfe_arr[i] = (entry - p_low) / entry * 100  # 空单：跌了是赚
                
        self.trades['MAE_Pct'] = mae_arr
        self.trades['MFE_Pct'] = mfe_arr

    def print_report(self, save_to_file=True):  # 保留之前新增的默认参数
        # 初始化输出内容存储列表
        report_content = []
        
        if self.trades.empty:
            output = "❌ 无交易记录，无法分析。"
            print(output)
            report_content.append(output)
            return

        # 构建报告内容，同时添加到列表和打印
        header1 = "\n" + "="*100
        header2 = "🔬 全维量化诊断报告 V5.0 (The God View)"
        header3 = "="*100
        print(header1)
        print(header2)
        print(header3)
        report_content.extend([header1, header2, header3])

        # ========== 新增：完整打印 self.trades (即 stats['_trades']) ==========
        trades_heading = "\n[0] 完整交易记录 (Full Trades Records - stats['_trades'])"
        trades_sep = "-" * 80
        print(trades_heading)
        print(trades_sep)
        report_content.extend([trades_heading, trades_sep])
        
        # 打印完整的 trades 数据框
        full_trades_str = self.trades.to_string(float_format="{:,.4f}".format)
        print(full_trades_str)
        report_content.append(full_trades_str)
        # =====================================================================

        # 1. 时序层级分析
        ts_heading = "\n[1] 时序层级分析 (Time Series)"
        ts_sep = "-" * 80
        print(ts_heading)
        print(ts_sep)
        report_content.extend([ts_heading, ts_sep])
        self._analyze_time_series(report_content)
        
        # 2. 连胜连败深度透视
        streak_heading = "\n[2] 连胜连败深度透视 (Streaks & Psychology)"
        streak_sep = "-" * 80
        print(streak_heading)
        print(streak_sep)
        report_content.extend([streak_heading, streak_sep])
        self._analyze_streaks_deep(report_content)

        # 3. 交易结构与持仓
        struct_heading = "\n[3] 交易结构与持仓 (Structure)"
        struct_sep = "-" * 80
        print(struct_heading)
        print(struct_sep)
        report_content.extend([struct_heading, struct_sep])
        self._analyze_structure(report_content)
        
        # 4. 亏损深度透视
        forensic_heading = "\n[4] 惨案现场取证 (Top Losses Forensics)"
        forensic_sep = "-" * 80
        print(forensic_heading)
        print(forensic_sep)
        report_content.extend([forensic_heading, forensic_sep])
        self._analyze_forensics(report_content)
        
        footer = "="*100
        print(footer)
        report_content.append(footer)

        # 如果需要保存到文件，将报告内容写入同目录的 result.txt
        if save_to_file:
            with open("/root/a9quant/strategies/result.txt", "w", encoding="utf-8") as f:
                # 拼接所有内容，每行之间换行分隔
                f.write("\n".join(report_content))

    def _analyze_time_series(self, report_content=None):
        # 1.1 Yearly
        yearly = self.trades.groupby('Year').agg({
            'PnL': ['sum', 'count'],
            'ReturnPct': 'mean'
        })
        yearly.columns = ['总盈亏(U)', '交易次数', '平均收益率']
        yearly_str = "1.1 年度汇总:\n" + yearly.to_string(float_format="{:,.2f}".format)
        print(yearly_str)
        if report_content is not None:
            report_content.append(yearly_str)
        
        # 1.2 Monthly Matrix (Full)
        monthly_heading = "\n1.2 月度盈亏矩阵 (全量):"
        print(monthly_heading)
        if report_content is not None:
            report_content.append(monthly_heading)
        
        monthly = self.trades.groupby(['Year', 'Month'])['PnL'].sum()
        # Reindex to ensure all months exist
        years = self.trades['Year'].unique()
        months = range(1, 13)
        idx = pd.MultiIndex.from_product([years, months], names=['Year', 'Month'])
        monthly = monthly.reindex(idx, fill_value=0).unstack()
        monthly['Total'] = monthly.sum(axis=1)
        
        # Format: 0 -> - (cleaner look), numbers -> formatted
        monthly_str = monthly.applymap(lambda x: '-' if x==0 else f"{x:,.0f}").to_string()
        print(monthly_str)
        if report_content is not None:
            report_content.append(monthly_str)

        # 1.3 Daily Calendar (Last 10 active days)
        daily_heading = "\n1.3 日度盈亏日历 (最近10个交易日):"
        print(daily_heading)
        if report_content is not None:
            report_content.append(daily_heading)
        
        daily = self.trades.groupby('Date').agg({
            'PnL': 'sum',
            'Size': 'count' # count trades
        }).sort_index(ascending=False).head(10)
        daily.columns = ['当日盈亏(U)', '次数']
        daily['状态'] = daily['当日盈亏(U)'].apply(lambda x: '✅ 盈利' if x>0 else '❌ 亏损')
        daily_str = daily.to_string(float_format="{:,.2f}".format)
        print(daily_str)
        if report_content is not None:
            report_content.append(daily_str)

    def _analyze_streaks_deep(self, report_content=None):
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
        win_content = []
        if not wins.empty:
            max_win = wins.loc[wins['Count'].idxmax()]
            win_content.append(f"🔥 连胜分析:")
            win_content.append(f"   - 最大连胜次数: {max_win['Count']} 次")
            win_content.append(f"   - 连胜期间总赚: {max_win['Total_PnL']:,.2f} U (平均每单 {max_win['Avg_PnL']:,.2f} U)")
            win_content.append(f"   - 发生时间: {max_win['Start'].strftime('%Y-%m-%d')} ~ {max_win['End'].strftime('%Y-%m-%d')}")
        
        win_str = "\n".join(win_content)
        print(win_str)
        if report_content is not None:
            report_content.append(win_str)
        
        # Loss Analysis
        losses = df_streaks[df_streaks['Type'] == 'Loss']
        loss_content = []
        if not losses.empty:
            max_loss = losses.loc[losses['Count'].idxmax()]
            # Find the streak with worst total PnL (might not be the longest)
            worst_pnl_streak = losses.loc[losses['Total_PnL'].idxmin()]
            
            loss_content.append(f"\n❄️ 连败分析 (最长):")
            loss_content.append(f"   - 最大连败次数: {max_loss['Count']} 次")
            loss_content.append(f"   - 连败期间总亏: {max_loss['Total_PnL']:,.2f} U (平均每单 {max_loss['Avg_PnL']:,.2f} U)")
            
            if worst_pnl_streak['Count'] != max_loss['Count']:
                loss_content.append(f"\n❄️ 连败分析 (最痛 - 亏钱最多):")
                loss_content.append(f"   - 连败次数: {worst_pnl_streak['Count']} 次")
                loss_content.append(f"   - 期间总亏: {worst_pnl_streak['Total_PnL']:,.2f} U")
                loss_content.append(f"   - 发生时间: {worst_pnl_streak['Start'].strftime('%Y-%m-%d')} ~ {worst_pnl_streak['End'].strftime('%Y-%m-%d')}")
        
        loss_str = "\n".join(loss_content)
        print(loss_str)
        if report_content is not None:
            report_content.append(loss_str)

    def _analyze_structure(self, report_content=None):
        # Long/Short Stats
        struct_header = f"{'类型':<8} {'次数':<6} {'胜率%':<8} {'平均盈亏':<12} {'平均持仓':<10} {'盈亏比':<8}"
        print(struct_header)
        if report_content is not None:
            report_content.append(struct_header)
        
        dir_content = []
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
            
            dir_line = f"{direction:<8} {count:<6} {win_rate:<8.1f} {avg_pnl:<12.0f} {avg_dur:<10.1f} {rr:<8.2f}"
            dir_content.append(dir_line)
        
        dir_str = "\n".join(dir_content)
        print(dir_str)
        if report_content is not None:
            report_content.append(dir_str)

        # Holding Behavior
        win_dur = self.trades[self.trades['PnL']>0]['Duration_Hours'].mean()
        loss_dur = self.trades[self.trades['PnL']<=0]['Duration_Hours'].mean()
        
        hold_content = [
            f"\n⏳ 持仓行为:",
            f"   - 盈利单平均持仓: {win_dur:.1f}h",
            f"   - 亏损单平均持仓: {loss_dur:.1f}h"
        ]
        if loss_dur > win_dur * 2:
            hold_content.append("   ⚠️ 警告: 亏损单持仓时间显著长于盈利单 (死扛风险)")
        
        hold_str = "\n".join(hold_content)
        print(hold_str)
        if report_content is not None:
            report_content.append(hold_str)

    def _analyze_forensics(self, report_content=None):
        losses = self.trades[self.trades['PnL'] < 0].sort_values('PnL', ascending=True).head(50)
        if losses.empty:
            no_loss = "🎉 无亏损记录"
            print(no_loss)
            if report_content is not None:
                report_content.append(no_loss)
            return
            
        # Dynamic Indicators
        ind_cols = [c for c in self.trades.columns if any(x in c for x in ['SVD', 'Entropy', 'ADX', 'RSI', 'Strength', 'ATR'])]
        # Simplify names
        rename_map = {c: c.replace('Entry_', '').replace('_compute_', '')[:8] for c in ind_cols}
        
        forensic_header = f"{'Time':<16} {'Dir':<5} {'PnL(U)':<10} {'MAE%':<6} {'MFE%':<6} | {'Indicators'}"
        forensic_sep = "-" * 80
        print(forensic_header)
        print(forensic_sep)
        if report_content is not None:
            report_content.extend([forensic_header, forensic_sep])
        
        forensic_lines = []
        for _, row in losses.iterrows():
            t_str = row['EntryTime'].strftime('%y-%m-%d %H:%M')
            d_str = row['Direction']
            pnl_str = f"{row['PnL']:.0f}"
            mae_str = f"{row.get('MAE_Pct', 0):.1f}"
            mfe_str = f"{row.get('MFE_Pct', 0):.1f}"
            
            ind_str = " | ".join([f"{rename_map[c]}:{row[c]:.2f}" for c in ind_cols if pd.notnull(row[c])])
            
            line = f"{t_str:<16} {d_str:<5} {pnl_str:<10} {mae_str:<6} {mfe_str:<6} | {ind_str}"
            forensic_lines.append(line)
        
        forensic_str = "\n".join(forensic_lines)
        print(forensic_str)
        if report_content is not None:
            report_content.append(forensic_str)

    def format_with_chinese(self, initial_cash, commission):
        """
        原始数据+中文标注格式化方法
        保留所有原始字段，仅添加中文标注，不修改任何数值
        参数：
            initial_cash: 初始本金
            commission: 手续费率 (如 0.002 表示 0.2%)
        """
        # 通用格式化函数
        def safe_format(x):
            try:
                if x is None or pd.isna(x):
                    return "N/A"
                return f"{float(x):.2f}"
            except:
                return str(x) if x is not None else "N/A"
        
        # 1. 回测统计字段中文映射（包含所有字段，不省略）
        print("\n" + "="*60 + " 回测核心统计 (中文标注) " + "="*60)
        field_mapping = {
            'Start': '回测开始时间',
            'End': '回测结束时间',
            'Duration': '回测持续时长',
            'Exposure Time [%]': '持仓时间占比(%)',
            'Equity Final [$]': '最终账户权益(USDT)',
            'Equity Peak [$]': '账户权益峰值(USDT)',
            'Commissions [$]': '总手续费(USDT)',
            'Return [%]': '总收益率(%)',
            'Buy & Hold Return [%]': '买入持有收益率(%)',
            'Return (Ann.) [%]': '年化收益率(%)',
            'Volatility (Ann.) [%]': '年化波动率(%)',
            'CAGR [%]': '复合年增长率(%)',
            'Sharpe Ratio': '夏普比率',
            'Sortino Ratio': '索提诺比率',
            'Calmar Ratio': '卡尔玛比率',
            'Alpha [%]': '阿尔法系数(%)',
            'Beta': '贝塔系数',
            'Max. Drawdown [%]': '最大回撤(%)',
            'Avg. Drawdown [%]': '平均回撤(%)',
            'Max. Drawdown Duration': '最大回撤持续时长',
            'Avg. Drawdown Duration': '平均回撤持续时长',
            '# Trades': '总交易次数',
            'Win Rate [%]': '胜率(%)',
            'Best Trade [%]': '最佳单交易收益率(%)',
            'Worst Trade [%]': '最差单交易收益率(%)',
            'Avg. Trade [%]': '平均单交易收益率(%)',
            'Max. Trade Duration': '最长交易时长',
            'Avg. Trade Duration': '平均交易时长',
            'Profit Factor': '盈利因子',
            'Expectancy [%]': '预期收益率(%)',
            'SQN': '系统质量数',
            'Kelly Criterion': '凯利准则',
            '_strategy': '使用策略',
            '_equity_curve': '权益曲线',
            '_trades': '交易记录'
        }
        
        # 输出所有回测统计字段（包含所有原始字段，不省略）
        for eng_name in self.stats.keys():
            cn_name = field_mapping.get(eng_name, eng_name)  # 无映射则保留原字段名
            value = self.stats[eng_name]
            if isinstance(value, pd.Timedelta):
                print(f"{cn_name:<20}: {str(value).split('.')[0]}")
            elif isinstance(value, (int, float)) and '%' in cn_name:
                print(f"{cn_name:<20}: {safe_format(value)}%")
            elif isinstance(value, (int, float)):
                print(f"{cn_name:<20}: {safe_format(value)}")
            else:
                print(f"{cn_name:<20}: {value}")
        
        # 2. 交易记录中文处理（保留所有原始字段）
        print("\n" + "="*60 + " 交易记录 (中文标注) " + "="*60)
        trades = self.stats['_trades']
        if not trades.empty:
            trades_cn = trades.copy()
            # 交易字段中文映射（保留所有字段，仅标注已有字段）
            field_alias = {
                'Size': '仓位大小',
                'EntryBar': '开仓K线索引',
                'ExitBar': '平仓K线索引',
                'EntryPrice': '开仓价格',
                'ExitPrice': '平仓价格',
                'SL': '止损价',
                'TP': '止盈价',
                'PnL': '盈亏(USDT)',
                'ReturnPct': '盈亏百分比(%)',  # 适配backtesting.py原生字段
                'Commission': '手续费(USDT)',
                'EntryTime': '开仓时间(UTC)',
                'ExitTime': '平仓时间(UTC)',
                'Duration': '交易时长',
                'Tag': '交易标签',
                'Entry_BreakHigh': '开仓突破高价',
                'Exit_BreakHigh': '平仓突破高价',
                'Entry_BreakLow': '开仓突破低价',
                'Exit_BreakLow': '平仓突破低价'
            }
            
            # 仅重命名有映射的字段，保留其他所有原始字段
            rename_dict = {k:v for k,v in field_alias.items() if k in trades_cn.columns}
            trades_cn.rename(columns=rename_dict, inplace=True)
            
            # 计算缺失字段（仅补充，不修改原始数据）
            # 兼容backtesting.py原生的ReturnPct字段，避免重复计算
            if '盈亏百分比(%)' not in trades_cn.columns and '盈亏(USDT)' in trades_cn.columns:
                trades_cn['盈亏百分比(%)'] = np.where(
                    trades_cn['开仓价格'] != 0,
                    (trades_cn['盈亏(USDT)'] / (abs(trades_cn['仓位大小']) * trades_cn['开仓价格'])) * 100,
                    0
                )
            # 补充计算手续费（backtesting.py原生trades无此字段）
            if '手续费(USDT)' not in trades_cn.columns:
                trades_cn['手续费(USDT)'] = abs(trades_cn['仓位大小']) * trades_cn['开仓价格'] * commission
            
            # 时间转换（仅添加北京时间列，保留原始UTC时间列）
            for time_col in ['开仓时间(UTC)', '平仓时间(UTC)']:
                if time_col in trades_cn.columns:
                    cn_col = time_col.replace('(UTC)', '(北京时间)')
                    trades_cn[cn_col] = trades_cn[time_col].apply(
                        lambda x: (x + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S') if pd.notna(x) else 'N/A'
                    )
            
            # 输出完整交易记录（包含所有字段）
            print(trades_cn.to_string(index=True, float_format=lambda x: safe_format(x), na_rep="N/A"))
        else:
            print("⚠️ 无交易记录")
        
        # 3. 权益曲线信息（保留所有原始信息）
        print("\n" + "="*60 + " 权益曲线信息 " + "="*60)
        equity_curve = self.stats['_equity_curve']
        try:
            if not equity_curve.empty:
                print(f"权益曲线时间范围: {equity_curve.index.min()} ~ {equity_curve.index.max()}")
                print(f"初始权益: {safe_format(initial_cash)} USDT")
                print(f"最终权益: {safe_format(self.stats.get('Equity Final [$]'))} USDT")
                print(f"权益峰值: {safe_format(self.stats.get('Equity Peak [$]'))} USDT")
                # 补充谷值计算（backtesting.py原生stats无此字段）
                equity_min = equity_curve['Equity'].min() if 'Equity' in equity_curve.columns else 0
                print(f"权益谷值: {safe_format(equity_min)} USDT")
        except Exception as e:
            print(f"⚠️ 读取权益曲线数据出错: {str(e)}")
            print(f"初始权益: {safe_format(initial_cash)} USDT")
            print(f"最终权益: {safe_format(self.stats.get('Equity Final [$]', 0))} USDT")
            print(f"权益峰值: {safe_format(self.stats.get('Equity Peak [$]', 0))} USDT")