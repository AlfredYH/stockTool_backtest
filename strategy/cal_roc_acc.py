import pandas as pd
import numpy as np
from data_collect_akshare import collectfrom_aksh

class CalRocAcc:
    """
    计算并绘制特定周期变动率（ROC）的类。
    计算roc差以及多期累加值
    周期在创建实例时确定。
    """

    def __init__(self, data: pd.DataFrame, periodA: int = 1, periodB: int = 1, roll_period: int = 1):
        """
        初始化方法，传入原始数据和需要做差的两个ROC计算周期。

        参数:
        data (pd.DataFrame): 索引为日期，列为OHLCV等指标。
        periodA (int): 第一个ROC计算周期（N期）。
        periodB (int): 第二个ROC计算周期（M期）。
        """
        self.data = data
        self.periodA = periodA  # 将periodA保存为实例属性
        self.periodB = periodB  # 将periodB保存为实例属性
        self.roll_period = roll_period  # 将roll_period保存为实例属性
        # 预计算一次ROC数据并存储，避免重复计算
        # 这样写可以让后续调用更高效
        self.roc_data = self._calculate_acc_rocdiff_internal()

    def _calculate_acc_rocdiff_internal(self) -> pd.DataFrame:
        """内部方法，用于计算ROC并返回合并后的数据"""
        rocA = (self.data - self.data.shift(self.periodA)) / self.data.shift(self.periodA)
        rocB = (self.data - self.data.shift(self.periodB)) / self.data.shift(self.periodB)

        rocA = rocA.replace([np.inf, -np.inf], np.nan)
        rocB = rocB.replace([np.inf, -np.inf], np.nan)

        roc_diff = rocA - rocB
        acc_roc_diff = roc_diff.rolling(window=self.roll_period).sum()

        rocA.columns = [f"{col}_rocA_{self.periodA}" for col in rocA.columns]
        rocB.columns = [f"{col}_rocB_{self.periodB}" for col in rocB.columns]
        roc_diff.columns = [f"{col}_roc_diff_{self.periodA}_{self.periodB}" for col in roc_diff.columns]
        acc_roc_diff.columns = [f"{col}_acc_roc_diff_{self.periodA}_{self.periodB}_{self.roll_period}"
                                for col in acc_roc_diff.columns]

        # 合并原始数据和ROC数据
        df_list = [rocA, rocB, roc_diff, acc_roc_diff]
        data_all = pd.concat(df_list, axis=1)
        return data_all
    
    def get_roc_data(self) -> pd.DataFrame:
        """获取包含ROC值的完整DataFrame"""
        # 添加dropna方法，便于画图,mplfinance 是否可以处理nan？
        # plot_data = self.roc_data.dropna()
        # return plot_data
        return self.roc_data
    
if __name__ == "__main__":
    stock_data = collectfrom_aksh('sh688981', '20250220', '20251219')
    stock_calrocacc = CalRocAcc(stock_data, periodA=30, periodB=10, roll_period=15)
    stock_roc_acc = stock_calrocacc.get_roc_data()
    print(stock_roc_acc.head(5))
    print(stock_roc_acc.tail(5))
    print(stock_roc_acc.columns)
    print(f"DataFrame 的总行数: {len(stock_roc_acc)}")
    print(f"DataFrame 的形状 (行数, 列数): {stock_roc_acc.shape}")

"""
            open_rocA_30  high_rocA_30  low_rocA_30  ...  low_acc_roc_diff_30_10_15  close_acc_roc_diff_30_10_15  volume_acc_roc_diff_30_10_15
date                                                 ...
2025-12-15     -0.056148     -0.049755    -0.038707  ...                  -1.329506                    -1.400362                     -5.808559
2025-12-16     -0.070248     -0.075901    -0.073361  ...                  -1.319012                    -1.385961                     -5.610600
2025-12-17     -0.052010     -0.059455    -0.054678  ...                  -1.296736                    -1.382895                     -5.904073
2025-12-18     -0.074651     -0.081514    -0.068567  ...                  -1.314833                    -1.432380                     -6.188846
2025-12-19     -0.079151     -0.070060    -0.074150  ...                  -1.370130                    -1.467063                     -6.226829
[5 rows x  twelve columns]
列名: Index(['open_rocA_30', 'high_rocA_30', 'low_rocA_30', 'close_rocA_30',
       'volume_rocA_30', 'open_rocB_10', 'high_rocB_10', 'low_rocB_10',
       'close_rocB_10', 'volume_rocB_10', 'open_roc_diff_30_10',
       'high_roc_diff_30_10', 'low_roc_diff_30_10', 'close_roc_diff_30_10',
       'volume_roc_diff_30_10', 'open_acc_roc_diff_30_10_15',
       'high_acc_roc_diff_30_10_15', 'low_acc_roc_diff_30_10_15',
       'close_acc_roc_diff_30_10_15', 'volume_acc_roc_diff_30_10_15'],
      dtype='object')
"""