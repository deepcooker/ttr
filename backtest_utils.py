# /root/a9quant/strategies/backtest_utils.py
import sys
import os
from datetime import timedelta
import pandas as pd
import numpy as np
# 新增：导入DataFeed类
from core.datafeed import DataFeed

class BacktestUtils:
    """回测工具类：封装数据准备和中文标注通用逻辑"""
    
    @staticmethod
    def prepare_data(symbol, start_str, end_str, source_dir, target_timeframe='1m'):
        """
        生产版：通用数据加载准备
        :param source_dir: 数据所在目录
        :param target_timeframe: 目标策略周期 (默认 '1m', 可选 '1s', '5m' 等)
        """
        print(f"\n📂 加载数据 {symbol} {start_str} -> {end_str}")
        print(f"   🎯 目标周期: {target_timeframe}")
        print(f"   📦 数据源路径: {source_dir}")
        
        try:
            # 1. 构造DataFeed配置
            config = {
                'system': {
                    'data_dir': os.path.dirname(source_dir),  # 数据根目录
                    'parquet_subdir': os.path.basename(source_dir)  # parquet子目录
                }
            }
            data_feed = DataFeed(config)
            
            # 2. 转换日期格式
            start_date = pd.Timestamp(start_str).strftime("%Y%m%d")
            end_date = pd.Timestamp(end_str).strftime("%Y%m%d")
            
            # 3. 调用DataFeed加载数据
            # 🔥 关键：将 target_timeframe 传进去，DataFeed 会自动决定是读1m还是1s
            df_data = data_feed.load_data(symbol, start_date, end_date, timeframe=target_timeframe)
            
            if df_data is None:
                print("❌ DataFeed加载数据失败")
                return None
            
            # 4. 严格切片：仅保留请求的时间范围
            start_dt = pd.Timestamp(start_str).tz_localize('UTC')
            end_dt = pd.Timestamp(end_str).tz_localize('UTC') + timedelta(days=1) - timedelta(seconds=1)
            df_data = df_data.loc[start_dt:end_dt]
            
            # 5. 最终校验 (日志通用化)
            print(f"✅ 数据加载完成：")
            print(f"   - K线周期: {target_timeframe}")
            print(f"   - 数据行数: {len(df_data)}")
            print(f"   - 时间范围: {df_data.index.min()} ~ {df_data.index.max()}")
            
            if len(df_data) > 0:
                print(f"   - 首条 High: {df_data.iloc[0]['High']}")
                print(f"   - 末条 High: {df_data.iloc[-1]['High']}")
            
            return df_data

        except Exception as e:
            print(f"❌ 数据准备失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def format_with_chinese(stats, initial_cash, commission):
        """
        原始数据+中文标注格式化方法
        保留所有原始字段，仅添加中文标注，不修改任何数值
        参数：
            stats: 回测返回的stats对象
            initial_cash: 初始本金
            commission: 手续费率
        """
        # 通用格式化函数
        def safe_format(x):
            try:
                if x is None or pd.isna(x):
                    return "N/A"
                return f"{float(x):.2f}"
            except:
                return str(x) if x is not None else "N/A"
        
       
        
        # 2. 交易记录中文处理（保留所有原始字段）
        trades = stats['_trades']
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
                'PnL%': '盈亏百分比(%)',
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
            if '盈亏百分比(%)' not in trades_cn.columns and '盈亏(USDT)' in trades_cn.columns:
                trades_cn['盈亏百分比(%)'] = np.where(
                    trades_cn['开仓价格'] != 0,
                    (trades_cn['盈亏(USDT)'] / (abs(trades_cn['仓位大小']) * trades_cn['开仓价格'])) * 100,
                    0
                )
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
            
            
         # 1. 回测统计字段中文映射（包含所有字段，不省略）
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
        for eng_name in stats.keys():
            cn_name = field_mapping.get(eng_name, eng_name)  # 无映射则保留原字段名
            value = stats[eng_name]
            if isinstance(value, pd.Timedelta):
                print(f"{cn_name:<15}: {str(value).split('.')[0]}")
            elif isinstance(value, (int, float)) and '%' in cn_name:
                print(f"{cn_name:<15}: {safe_format(value)}%")
            elif isinstance(value, (int, float)):
                print(f"{cn_name:<15}: {safe_format(value)}")
            else:
                print(f"{cn_name:<15}: {value}")
        
        # 3. 权益曲线信息（保留所有原始信息）
        equity_curve = stats['_equity_curve']
        try:
            if not equity_curve.empty:
                print(f"权益曲线时间范围: {equity_curve.index.min()} ~ {equity_curve.index.max()}")
                print(f"初始权益: {safe_format(initial_cash)} USDT")
                print(f"最终权益: {safe_format(stats.get('Equity Final [$]'))} USDT")
                print(f"权益峰值: {safe_format(stats.get('Equity Peak [$]'))} USDT")
                print(f"权益谷值: {safe_format(stats.get('Equity Min [$]', 0))} USDT")
                # 输出权益曲线所有原始数据
                #print("权益曲线原始数据:")
                #print(equity_curve)
        except Exception as e:
            print(f"⚠️ 读取权益曲线数据出错: {str(e)}")
            print(f"初始权益: {safe_format(initial_cash)} USDT")
            print(f"最终权益: {safe_format(stats.get('Equity Final [$]', 0))} USDT")
            print(f"权益峰值: {safe_format(stats.get('Equity Peak [$]', 0))} USDT")