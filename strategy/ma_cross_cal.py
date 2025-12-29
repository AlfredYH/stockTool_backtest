"""
writer: Google Gemini 3 Pro
优化后的健壮数据提供器
修改点：
1. 移除了低效的 for 循环，使用 pandas 向量化操作生成信号。
2. 简化了索引对齐逻辑。
3. 增加了对原始数据含 NaN 的简单处理建议（这里采用 ffill 保证 convolve 稳定性）。
4. 优化了内存使用（类型转换）。

一个规范的健壮的数据提供器，实现：
类的输入为价格序列，类型为dataframe，不想用pd.series, 偷懒了
还有两个周期long和short, 类型为int
用之前的convolve实现均线计算，
输出：
Index: 与输入的 prices 序列对齐的整数索引。
price 列: 包含原始价格数据。
ma_short 列: 包含短期均线数据。
ma_long 列: 包含长期均线数据。
signal 列: 这是最重要的信号列。1 表示金叉，-1 表示死叉，0 表示无信号。这种格式是量化交易策略中常用的。
cross_price 列: 记录发生交叉时的价格。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MovingAverageCrossover:
    """
    一个高效、健壮的移动平均线交叉信号数据提供器。
    采用向量化计算替代循环，大幅提升大数据量下的处理速度。
    """
    
    def __init__(self, price_dataframe: pd.DataFrame,
                 short_window: int, long_window: int, price_column: str = 'price'):
        """
        初始化并执行计算。
        """
        self._validate_inputs(price_dataframe, short_window, long_window, price_column)
        
        self.short_window = short_window
        self.long_window = long_window
        self.price_column = price_column
        
        # 复制数据，重置索引以确保 align 对齐安全，最后再还原
        # 很多时候传入的 DF 索引可能不连续，重置索引可以保证 numpy 计算时的绝对对齐
        self._original_index = price_dataframe.index
        self._df = price_dataframe[[price_column]].copy().reset_index(drop=True)
        
        # 简单的预处理：填充空值，防止 convolve 结果大面积坍塌
        # 注意：这里假设前向填充是合理的，实际业务中需根据需求调整
        if self._df[self.price_column].isnull().any():
            self._df[self.price_column] = self._df[self.price_column].ffill()

        self._calculate_all()

    def _validate_inputs(self, df, short_w, long_w, col):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if not isinstance(short_w, int) or not isinstance(long_w, int):
            raise TypeError("Windows must be integers.")
        if short_w >= long_w:
            raise ValueError("Short window must be smaller than long window.")
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        if len(df) < long_w:
            raise ValueError("Data length is shorter than the long window.")

    def _calculate_ma_convolve(self, prices, window):
        """
        使用 np.convolve 计算均线 (Constraints requirement).
        """
        weights = np.ones(window) / window
        # mode='valid': 只计算完全覆盖的窗口
        ma_valid = np.convolve(prices, weights, mode='valid')
        
        # 为了与原始序列对齐，需要在前面填充 (window - 1) 个 NaN
        # 使用 np.full 填充 NaN，效率高
        padding = np.full(window - 1, np.nan)
        return np.concatenate([padding, ma_valid])

    def _calculate_all(self):
        """
        核心计算逻辑：向量化实现
        """
        prices = self._df[self.price_column].values
        
        # 1. 计算均线 (Numpy 操作)
        ma_short = self._calculate_ma_convolve(prices, self.short_window)
        ma_long = self._calculate_ma_convolve(prices, self.long_window)
        
        # 将结果存入 DF，方便后续使用 Pandas 的 shift 操作
        self._df['ma_short'] = ma_short
        self._df['ma_long'] = ma_long
        
        # 2. 向量化计算信号 (Pandas 操作)
        # 逻辑：
        # 金叉 (Golden Cross): t-1 时刻 短线 < 长线 AND t 时刻 短线 > 长线
        # 死叉 (Death Cross) : t-1 时刻 短线 > 长线 AND t 时刻 短线 < 长线
        
        # 获取 Series 用于比较
        s = self._df['ma_short']
        l = self._df['ma_long']
        
        # shift(1) 代表上一时刻的值
        prev_s = s.shift(1)
        prev_l = l.shift(1)
        
        # 布尔掩码
        golden_cross_mask = (prev_s <= prev_l) & (s > l)
        death_cross_mask = (prev_s >= prev_l) & (s < l)
        
        # 3. 初始化结果列
        self._df['signal'] = 0
        self._df['cross_price'] = np.nan
        
        # 4. 应用掩码 (利用 numpy where 或 loc 赋值，速度极快)
        # 赋值金叉
        self._df.loc[golden_cross_mask, 'signal'] = 1
        self._df.loc[golden_cross_mask, 'cross_price'] = \
            self._df.loc[golden_cross_mask, self.price_column]
        
        # 赋值死叉
        self._df.loc[death_cross_mask, 'signal'] = -1
        self._df.loc[death_cross_mask, 'cross_price'] = \
            self._df.loc[death_cross_mask, self.price_column]

        # 优化 signal 列的数据类型为整数 (可选)
        self._df['signal'] = self._df['signal'].astype(int)

    def get_dataframe(self):
        """
        返回最终结果，索引还原为原始输入的索引。
        """
        result = self._df.copy()
        result.index = self._original_index
        return result

    def plot(self, ax=None):
        """
        可视化
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 7))
        
        # 获取还原索引后的数据用于绘图
        plot_df = self.get_dataframe()
        
        ax.plot(plot_df.index, plot_df[self.price_column], label='Price', color='gray', alpha=0.5)
        ax.plot(plot_df.index, plot_df['ma_short'],
                label=f'MA{self.short_window}', color='#1f77b4', linewidth=1.5)
        ax.plot(plot_df.index, plot_df['ma_long'],
                label=f'MA{self.long_window}', color='#ff7f0e', linewidth=1.5)
        
        # 提取信号点
        buy_sig = plot_df[plot_df['signal'] == 1]
        sell_sig = plot_df[plot_df['signal'] == -1]
        
        if not buy_sig.empty:
            ax.scatter(buy_sig.index, buy_sig['cross_price'],
                       marker='^', color='green', s=100, label='Golden Cross', zorder=5)
        if not sell_sig.empty:
            ax.scatter(sell_sig.index, sell_sig['cross_price'],
                       marker='v', color='red', s=100, label='Death Cross', zorder=5)
            
        ax.set_title('Moving Average Crossover Strategy (Vectorized)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if ax is None:
            plt.show()

# --- 测试代码 ---
if __name__ == "__main__":
    # 1. 模拟数据生成
    np.random.seed(2023)
    dates = pd.date_range(start='2022-01-01', periods=500, freq='D')
    # 制造一些明显的波动趋势
    x = np.linspace(0, 4*np.pi, 500)
    trend = 100 + x * 5 + np.sin(x) * 20
    noise = np.random.normal(0, 2, 500)
    prices = trend + noise
    
    df_input = pd.DataFrame({'close': prices}, index=dates)
    
    print(">>> 初始化策略数据提供器...")
    try:
        # 2. 实例化 (使用 'close' 列作为价格)
        provider = MovingAverageCrossover(df_input, short_window=20, long_window=60, price_column='close')
        
        # 3. 获取结果
        res_df = provider.get_dataframe()
        
        # 4. 打印信号统计
        print("\n>>> 数据预览 (Tail):")
        print(res_df.tail())
        
        print("\n>>> 信号统计:")
        print(res_df['signal'].value_counts())
        
        buy_signals = res_df[res_df['signal'] == 1]
        print(f"\n>>> 发现 {len(buy_signals)} 个金叉信号:")
        if not buy_signals.empty:
            print(buy_signals[['close', 'ma_short', 'ma_long', 'cross_price']])

        # 5. 绘图
        provider.plot()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
