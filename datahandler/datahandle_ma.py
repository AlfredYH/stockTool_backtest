import pandas as pd
import numpy as np
from data_collect_akshare import collectfrom_aksh


class DataHandleMa():

    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date

        print(f"Initialized DataHandleMa with periods.start_date={self.start_date}, end_date={self.end_date}")
    
    def get_all_Astocks(self):
        """
        定义一个可以遍历股票池并删选出目标股票的函数
        遍历我已经准备好的股票池CSV文件，读取其中的股票代码和名称
        """
        try:
            all_stocks = pd.read_csv(
            'data/20250723_215643_A_board_stocklist.csv',
            # 默认逗号sep='\t',
            # index_col='序号',
            usecols=['代码', '名称'],
            dtype={'代码': str, '名称': str}
            )
            # print("列名：", all_stocks.columns.tolist())
            return all_stocks
        except ValueError as e:
            print(f"读取股票池失败：{e}")
            return pd.DataFrame()  # 返回空DataFrame

    def cal_ma(self, stock_his: pd.DataFrame, period_list: np.ndarray) -> pd.DataFrame:
        """
        计算均线数据
        """
        for period in period_list:
            try:
                name = f"MA{period}"
                stock_his[name] = stock_his['close'].rolling(window=period).mean()
            except Exception as e:
                print(f"MA计算出错：{e}")
                continue
        return stock_his
    
    def get_stocks_his(self):
        all_stocks = self.get_all_Astocks()
        stock_histories = {}
        for i in range(len(all_stocks)):
            stock_code = all_stocks.iloc[i]['代码']
            stock_name = all_stocks.iloc[i]['名称']
            print(f"\n正在处理：{stock_name}（{stock_code}）")

            try:
                stock_his = collectfrom_aksh(stock_code, self.start_date, self.end_date)
                if stock_his is None or stock_his.empty:
                    print("获取数据失败（空数据）")
                    continue
                stock_his_ma = self.cal_ma(stock_his)
                stock_histories[stock_code] = stock_his_ma
                print(f"成功获取并计算均线数据")
            except Exception as e:
                print(f"处理股票出错：{e}")
                continue
        return stock_histories
    
    def df_to_csv(self, df: pd.DataFrame, file_path: str):
        file_name = "target_stocks"
        df.to_csv('data/' + file_name + '.csv')
        print(f"DataFrame saved to data文件夹下的 {file_name}.csv")

if __name__ == "__main__":
    data_handler = DataHandleMa(
        period_list=np.array([5, 10, 20, 30, 60]),
        start_date='20240101',
        end_date='20250723'
    )
    stock_histories = data_handler.get_stocks_his()
    # 示例：保存某只股票的均线数据到CSV
    if '600970' in stock_histories:
        data_handler.df_to_csv(stock_histories['600970'], 'data/600970_ma_data.csv')