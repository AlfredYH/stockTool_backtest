"""
变动率（ROC）计算与绘图模块
(period作为类属性的版本)
"""

import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt


class CalRoc:
    """
    计算并绘制特定周期变动率（ROC）的类。
    周期在创建实例时确定。
    """

    def __init__(self, data: pd.DataFrame, period: int = 1):
        """
        初始化方法，传入原始数据和ROC计算周期。

        参数:
        data (pd.DataFrame): 索引为日期，列为OHLCV等指标。
        period (int): ROC计算周期（N期），此实例的所有操作都将使用此周期。
        """
        self.data = data
        self.period = period  # 将period保存为实例属性
        # 预计算一次ROC数据并存储，避免重复计算
        # 这样写可以让后续调用更高效
        self.roc_data = self._calculate_roc_internal()

    def _calculate_roc_internal(self) -> pd.DataFrame:
        """内部方法，用于计算ROC并返回合并后的数据"""
        roc = (self.data - self.data.shift(self.period)) / self.data.shift(self.period)
        roc = roc.replace([np.inf, -np.inf], np.nan)
        roc.columns = [f"{col}_roc_{self.period}" for col in roc.columns]
        
        # 合并原始数据和ROC数据
        data_all = pd.merge(
            self.data,
            roc,
            left_index=True,
            right_index=True,
            how='left'
        )
        return data_all

    """
    这里触及了面向对象编程（OOP）中一个核心的设计原则：封装（Encapsulation）
    以下的函数很重要，
    这里的方法默认只提供读取功能，用户无法通过它来修改self.roc_data,
    保证了类的内部状态不被意外修改。
    """
    def get_roc_data(self) -> pd.DataFrame:
        """获取包含ROC值的完整DataFrame"""
        return self.roc_data

    def plot_roc(self, column: str, **kwargs):
        """
        绘制指定列的ROC指标图。

        参数:
        column (str): 需要绘制ROC的原始列名（如 'Close', 'Volume'）。
        **kwargs: 传递给 mpf.plot 的其他参数。
        """
        roc_column_name = f"{column}_roc_{self.period}"
        
        if roc_column_name not in self.roc_data.columns:
            raise ValueError(f"错误：无法找到列 '{roc_column_name}'。请确保 'column' 参数正确。")

        plot_data = self.roc_data.dropna()

        if plot_data.empty:
            print(f"警告：由于前{self.period}期数据不足，没有足够的有效数据用于绘图。")
            return

        apds = mpf.make_addplot(plot_data[roc_column_name], panel=1, color='g', ylabel=f'ROC({self.period})')
        
        mpf.plot(plot_data, addplot=apds, **kwargs)


# ------------------- 测试示例 -------------------
if __name__ == "__main__":
    dates = pd.date_range(start="2025-01-01", periods=100, freq="D")
    price = pd.Series([100 + i*0.5 + np.random.randn()*2 for i in range(100)], index=dates)
    volume = pd.Series([1000 + i*10 + np.random.randint(-100, 100) for i in range(100)], index=dates)
    test_data = pd.DataFrame({
        "Open": price - 2, "High": price + 2, "Low": price - 2, "Close": price, "Volume": volume
    })

    # 创建一个专门用于30期ROC分析的实例
    print("正在创建30期ROC分析器...")
    roc_30_analyzer = CalRoc(test_data, period=30)
    
    # 获取30期ROC数据
    roc_data_30 = roc_30_analyzer.get_roc_data()
    print("30期ROC数据（后5行）：")
    print(roc_data_30[['Close_roc_30', 'Volume_roc_30']].tail())
    
    # 绘制30期成交量ROC图
    print("\n正在绘制30期成交量ROC图...")
    roc_30_analyzer.plot_roc(
        column='Volume', 
        type='candle',
        volume=True,
        title='price, volume and roc(30)',
        figratio=(16, 10)
    )

    # 如果需要分析不同周期，需要创建新的实例
    print("\n\n正在创建10期ROC分析器...")
    roc_10_analyzer = CalRoc(test_data, period=10)
    roc_10_analyzer.plot_roc(column='Close', type='line', title='close and roc(10)', figratio=(16, 8))