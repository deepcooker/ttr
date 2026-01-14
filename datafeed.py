# 文件路径: /root/a9quant/core/datafeed.py
import pandas as pd
import os
import glob
from functools import lru_cache
from datetime import datetime, timedelta

@lru_cache(maxsize=8) 
def _cached_load_data(data_dir_root, symbol, start_date, end_date, timeframe):
    print(f"📂 [DataFeed] 磁盘读取(未命中缓存): {symbol} | {start_date} -> {end_date} | 目标周期: {timeframe}")
    
    try:
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        # 多读一天，交给上层切片，保证UTC 24小时完整
        adjusted_end = end + timedelta(days=1)
        date_list = [start + timedelta(days=x) for x in range((adjusted_end - start).days + 1)]
    except ValueError:
        print("❌ 日期格式错误，请使用 YYYYMMDD")
        return None

    df_list = []
    source_freq_is_1m = False # 标记源数据频率：True=分钟线, False=秒线

    for dt in date_list:
        dt_str = dt.strftime("%Y%m%d")
        
        # 1. 构造两种可能的文件名
        path_1m = os.path.join(data_dir_root, f"{symbol}_{dt_str}_1m_UTC.parquet")
        path_1s = os.path.join(data_dir_root, f"{symbol}_{dt_str}_1s_UTC.parquet")
        
        target_path = None
        
        # ========== 🔥 核心逻辑修正：默认分K，强制秒K ==========
        
        if timeframe == '1s':
            # 🔴 [强制模式] 如果请求的是 1s 数据，必须强制找 1s 文件
            # 绝对不能加载 1m 文件，因为无法从分K变出秒K
            if os.path.exists(path_1s):
                target_path = path_1s
                source_freq_is_1m = False
            # 如果没找到 1s，target_path 保持 None，跳过此日期
            
        else:
            # 🟢 [常规模式] 如果请求的是 1m 或更高周期 (5m, 1h...)
            # 优先找 [1m] (默认标准，速度最快)
            if os.path.exists(path_1m):
                target_path = path_1m
                source_freq_is_1m = True
            # 其次找 [1s] (作为可选的兜底/高精度源)
            elif os.path.exists(path_1s):
                target_path = path_1s
                source_freq_is_1m = False
        # ====================================================

        if target_path:
            try:
                # 只读取需要的列，提升速度
                df_day = pd.read_parquet(target_path, columns=['open', 'high', 'low', 'close', 'volume'])
                df_list.append(df_day)
            except Exception as e:
                print(f"⚠️ 读取文件失败 {target_path}: {e}")

    if not df_list:
        print(f"❌ 未找到有效数据 (Symbol: {symbol}, 请求Timeframe: {timeframe})")
        print(f"   搜索路径: {data_dir_root}")
        return None

    # 3. 合并与时区处理
    full_df = pd.concat(df_list)
    
    if full_df.index.tz is None:
        full_df.index = full_df.index.tz_localize('UTC')
    else:
        full_df.index = full_df.index.tz_convert('UTC')

    if not full_df.index.is_monotonic_increasing:
        full_df.sort_index(inplace=True)
        
    full_df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    }, inplace=True)

    # 4. 智能降采样 (Smart Resampling)
    need_resample = True
    
    # 情况A: 请求1s，源数据是1s -> 不需处理
    if timeframe == '1s' and not source_freq_is_1m:
        need_resample = False
        
    # 情况B: 请求1m，源数据是1m -> 不需处理 (这是最常见的 "默认分K" 路径)
    elif timeframe in ['1m', '1min'] and source_freq_is_1m:
        need_resample = False

    if need_resample:
        # print(f"   🔄 Resample: 源数据({'1m' if source_freq_is_1m else '1s'}) -> {timeframe}")
        rule_map = {
            '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min',
            '30m': '30min', '45m': '45min', 
            '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', 
            '8h': '8H', '12h': '12H', '1d': '1D', '1w': '1W'
        }
        rule = rule_map.get(timeframe, timeframe)
        
        agg_dict = {
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }
        try:
            full_df = full_df.resample(rule).agg(agg_dict)
            full_df.dropna(inplace=True)
        except Exception as e:
            print(f"❌ 降采样失败 ({timeframe}): {e}")
            return None
            
    return full_df

class DataFeed:
    def __init__(self, config):
        self.data_root = config['system']['data_dir']
        self.subdir = config['system']['parquet_subdir']
        self.full_dir = os.path.join(self.data_root, self.subdir)
        
    # 🔥 这里的 timeframe 参数传给缓存函数，决定加载逻辑
    def load_data(self, symbol: str, start_date: str, end_date: str, timeframe: str = '1m'):
        return _cached_load_data(self.full_dir, symbol, start_date, end_date, timeframe)

    def check_integrity(self):
        files = glob.glob(os.path.join(self.full_dir, "*.parquet"))
        return len(files)