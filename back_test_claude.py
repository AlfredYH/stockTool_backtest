import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class BacktestFramework:
    """
    量化交易回测框架
    
    参数:
    - df: DataFrame, 包含'open', 'high', 'low', 'close', 'volume'的股票行情数据
    - hold: Series, 索引为日期，值为-1(卖出)、0(持有)、1(买入)
    - initial_capital: float, 初始资金，默认100000
    - commission: float, 手续费率，默认0.0003
    - slippage: float, 滑点率，默认0.001
    """
    
    def __init__(self, df, hold, initial_capital=100000, commission=0.0003, slippage=0.001):
        self.df = df.copy()
        self.hold = hold.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # 确保索引对齐
        self.df.index = pd.to_datetime(self.df.index)
        self.hold.index = pd.to_datetime(self.hold.index)
        
        # 初始化结果变量
        self.portfolio = None
        self.trades = None
        self.metrics = {}
        
    def run_backtest(self):
        """
        执行回测主函数
        """
        print("=" * 60)
        print("开始执行回测...")
        print("=" * 60)
        
        # 初始化账户
        cash = self.initial_capital
        position = 0  # 持仓数量
        portfolio_value = []
        cash_list = []
        position_list = []
        
        # 交易记录
        trades = []
        
        # 遍历每一天
        for date in self.df.index:
            if date not in self.hold.index:
                # 如果该日期没有交易信号，保持现状
                signal = 0
            else:
                signal = self.hold[date]
            
            # 获取当日价格（使用开盘价交易）
            price = self.df.loc[date, 'open']
            
            # 处理交易信号
            if signal == 1 and position == 0:  # 买入信号且当前空仓
                # 计算可买入数量（考虑滑点和手续费）
                effective_price = price * (1 + self.slippage)
                shares = int(cash / (effective_price * (1 + self.commission)))
                
                if shares > 0:
                    cost = shares * effective_price * (1 + self.commission)
                    cash -= cost
                    position = shares
                    
                    trades.append({
                        'date': date,
                        'action': 'BUY',
                        'price': price,
                        'shares': shares,
                        'cost': cost,
                        'cash': cash
                    })
                    
            elif signal == -1 and position > 0:  # 卖出信号且当前持仓
                # 卖出全部持仓（考虑滑点和手续费）
                effective_price = price * (1 - self.slippage)
                proceeds = position * effective_price * (1 - self.commission)
                cash += proceeds
                
                trades.append({
                    'date': date,
                    'action': 'SELL',
                    'price': price,
                    'shares': position,
                    'proceeds': proceeds,
                    'cash': cash
                })
                
                position = 0
            
            # 计算当日总资产（使用收盘价计算持仓市值）
            close_price = self.df.loc[date, 'close']
            total_value = cash + position * close_price
            
            portfolio_value.append(total_value)
            cash_list.append(cash)
            position_list.append(position * close_price)
        
        # 构建投资组合DataFrame
        self.portfolio = pd.DataFrame({
            'total_value': portfolio_value,
            'cash': cash_list,
            'position_value': position_list
        }, index=self.df.index)
        
        # 构建交易记录DataFrame
        if trades:
            self.trades = pd.DataFrame(trades)
        else:
            self.trades = pd.DataFrame()
            
        print(f"回测完成！共执行 {len(trades)} 笔交易")
        print("=" * 60)
        
        return self.portfolio
    
    def calculate_metrics(self):
        """
        计算策略评价指标
        """
        if self.portfolio is None:
            raise ValueError("请先运行回测: run_backtest()")
        
        # 计算收益率
        returns = self.portfolio['total_value'].pct_change().dropna()
        
        # 1. 总收益率
        total_return = (self.portfolio['total_value'].iloc[-1] / self.initial_capital - 1) * 100
        
        # 2. 年化收益率
        days = (self.portfolio.index[-1] - self.portfolio.index[0]).days
        annual_return = ((self.portfolio['total_value'].iloc[-1] / self.initial_capital) ** (365 / days) - 1) * 100
        
        # 3. 最大回撤
        cummax = self.portfolio['total_value'].cummax()
        drawdown = (self.portfolio['total_value'] - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        # 4. 夏普比率（假设无风险利率为3%）
        risk_free_rate = 0.03
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() != 0 else 0
        
        # 5. 波动率
        volatility = returns.std() * np.sqrt(252) * 100
        
        # 6. 胜率
        if len(self.trades) > 0:
            buy_trades = self.trades[self.trades['action'] == 'BUY'].copy()
            sell_trades = self.trades[self.trades['action'] == 'SELL'].copy()
            
            if len(buy_trades) > 0 and len(sell_trades) > 0:
                # 配对买卖交易
                wins = 0
                total_trades = min(len(buy_trades), len(sell_trades))
                
                for i in range(total_trades):
                    buy_price = buy_trades.iloc[i]['price']
                    sell_price = sell_trades.iloc[i]['price']
                    if sell_price > buy_price:
                        wins += 1
                
                win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            else:
                win_rate = 0
        else:
            win_rate = 0
        
        # 7. 盈亏比
        if len(returns) > 0:
            gains = returns[returns > 0]
            losses = returns[returns < 0]
            profit_loss_ratio = abs(gains.mean() / losses.mean()) if len(losses) > 0 and losses.mean() != 0 else 0
        else:
            profit_loss_ratio = 0
        
        # 8. Calmar比率（年化收益率/最大回撤）
        calmar_ratio = abs(annual_return / max_drawdown) if max_drawdown != 0 else 0
        
        # 9. Sortino比率
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_std if downside_std != 0 else 0
        
        # 10. IC（信息系数）- 交易信号与收益率的相关性
        if len(self.hold) > 0:
            # 对齐hold和returns
            aligned_data = pd.DataFrame({
                'signal': self.hold,
                'return': self.df['close'].pct_change()
            }).dropna()
            
            if len(aligned_data) > 0:
                ic = aligned_data['signal'].corr(aligned_data['return'])
            else:
                ic = 0
        else:
            ic = 0
        
        # 11. 交易次数
        num_trades = len(self.trades)
        
        # 12. 持仓天数
        holding_days = (self.portfolio['position_value'] > 0).sum()
        
        # 保存指标
        self.metrics = {
            '总收益率(%)': round(total_return, 2),
            '年化收益率(%)': round(annual_return, 2),
            '最大回撤(%)': round(max_drawdown, 2),
            '夏普比率': round(sharpe_ratio, 2),
            '波动率(%)': round(volatility, 2),
            '胜率(%)': round(win_rate, 2),
            '盈亏比': round(profit_loss_ratio, 2),
            'Calmar比率': round(calmar_ratio, 2),
            'Sortino比率': round(sortino_ratio, 2),
            'IC(信息系数)': round(ic, 4),
            '交易次数': num_trades,
            '持仓天数': holding_days,
            '初始资金': self.initial_capital,
            '最终资金': round(self.portfolio['total_value'].iloc[-1], 2)
        }
        
        return self.metrics
    
    def print_metrics(self):
        """
        打印策略评价指标
        """
        if not self.metrics:
            self.calculate_metrics()
        
        print("\n" + "=" * 60)
        print("策略评价指标".center(56))
        print("=" * 60)
        
        for key, value in self.metrics.items():
            print(f"{key:20s}: {value}")
        
        print("=" * 60 + "\n")
    
    def plot_kline(self, figsize=(16, 10)):
        """
        绘制K线图，包含成交量和买卖信号标注
        """
        if self.trades is None or self.portfolio is None:
            raise ValueError("请先运行回测: run_backtest()")
        
        # 准备买卖信号数据
        buy_signals = []
        sell_signals = []
        
        if len(self.trades) > 0:
            buy_trades = self.trades[self.trades['action'] == 'BUY']
            sell_trades = self.trades[self.trades['action'] == 'SELL']
            
            for _, trade in buy_trades.iterrows():
                buy_signals.append((trade['date'], trade['price']))
            
            for _, trade in sell_trades.iterrows():
                sell_signals.append((trade['date'], trade['price']))
        
        # 创建图形
        fig = plt.figure(figsize=figsize)
        
        # 使用mplfinance绘制K线图
        # 准备数据
        df_plot = self.df.copy()
        #print("df_plot 的所有列名：", df_plot.columns.tolist())
        #print(df_plot)
        #df_plot.columns = [col.capitalize() for col in df_plot.columns]
        
        # 创建附加图（成交量）
        apds = [
            mpf.make_addplot(df_plot['volume'], panel=1, type='bar', color='gray', alpha=0.5)
        ]
        
        # 设置样式
        mc = mpf.make_marketcolors(up='red', down='green', edge='inherit', wick='inherit', volume='inherit')
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=False)
        
        # 绘制K线图
        fig, axes = mpf.plot(df_plot, type='candle', style=s, addplot=apds,
                             volume=True, panel_ratios=(3, 1),
                             title='股票K线图与交易信号',
                             ylabel='价格', ylabel_lower='成交量',
                             figsize=figsize, returnfig=True)
        
        # 在K线图上标注买卖信号
        ax1 = axes[0]  # 主图
        
        # 标注买入信号
        for date, price in buy_signals:
            if date in df_plot.index:
                idx = df_plot.index.get_loc(date)
                ax1.annotate('', xy=(idx, price * 0.98), xytext=(idx, price * 0.93),
                           arrowprops=dict(arrowstyle='->', color='red', lw=2))
                ax1.text(idx, price * 0.92, 'B', color='red', fontsize=10, ha='center', weight='bold')
        
        # 标注卖出信号
        for date, price in sell_signals:
            if date in df_plot.index:
                idx = df_plot.index.get_loc(date)
                ax1.annotate('', xy=(idx, price * 1.02), xytext=(idx, price * 1.07),
                           arrowprops=dict(arrowstyle='->', color='green', lw=2))
                ax1.text(idx, price * 1.08, 'S', color='green', fontsize=10, ha='center', weight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print("K线图绘制完成！")
    
    def plot_portfolio(self, figsize=(16, 10)):
        """
        绘制资产、现金时序图和回撤曲线
        """
        if self.portfolio is None:
            raise ValueError("请先运行回测: run_backtest()")
        
        # 计算回撤
        cummax = self.portfolio['total_value'].cummax()
        drawdown = (self.portfolio['total_value'] - cummax) / cummax * 100
        
        # 创建子图
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # 第一个子图：总资产、现金、持仓市值
        ax1 = axes[0]
        ax1.plot(self.portfolio.index, self.portfolio['total_value'], 
                label='总资产', linewidth=2, color='blue')
        ax1.plot(self.portfolio.index, self.portfolio['cash'], 
                label='现金', linewidth=1.5, color='green', linestyle='--')
        ax1.plot(self.portfolio.index, self.portfolio['position_value'], 
                label='持仓市值', linewidth=1.5, color='red', linestyle='--')
        ax1.set_ylabel('资产金额', fontsize=12)
        ax1.set_title('投资组合资产变化', fontsize=14, weight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # 第二个子图：收益率曲线
        ax2 = axes[1]
        returns_cum = (self.portfolio['total_value'] / self.initial_capital - 1) * 100
        ax2.plot(self.portfolio.index, returns_cum, 
                linewidth=2, color='purple', label='累计收益率')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('收益率 (%)', fontsize=12)
        ax2.set_title('累计收益率曲线', fontsize=14, weight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # 第三个子图：回撤曲线
        ax3 = axes[2]
        ax3.fill_between(self.portfolio.index, drawdown, 0, 
                        color='red', alpha=0.3, label='回撤')
        ax3.plot(self.portfolio.index, drawdown, 
                linewidth=1.5, color='darkred')
        ax3.set_ylabel('回撤 (%)', fontsize=12)
        ax3.set_xlabel('日期', fontsize=12)
        ax3.set_title('回撤曲线', fontsize=14, weight='bold')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        print("资产时序图绘制完成！")
    
    def get_trade_details(self):
        """
        获取详细交易记录
        """
        if self.trades is None or len(self.trades) == 0:
            print("没有交易记录")
            return None
        
        print("\n" + "=" * 80)
        print("交易明细".center(76))
        print("=" * 80)
        print(self.trades.to_string(index=False))
        print("=" * 80 + "\n")
        
        return self.trades


def generate_test_data(start_date='2020-01-01', end_date='2023-12-31', 
                       initial_price=100, volatility=0.02):
    """
    生成测试用的股票数据和交易信号
    
    参数:
    - start_date: 开始日期
    - end_date: 结束日期
    - initial_price: 初始价格
    - volatility: 波动率
    
    返回:
    - df: 股票行情数据
    - hold: 交易信号序列
    """
    print("=" * 60)
    print("生成测试数据...")
    print("=" * 60)
    
    # 生成日期序列（仅工作日）
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n = len(dates)
    
    # 生成价格数据（使用几何布朗运动）
    np.random.seed(42)
    
    # 生成收盘价
    returns = np.random.normal(0.0005, volatility, n)
    price = initial_price * np.exp(np.cumsum(returns))
    
    # 生成开高低收
    close = price
    open_price = close * (1 + np.random.normal(0, 0.005, n))
    high = np.maximum(open_price, close) * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = np.minimum(open_price, close) * (1 - np.abs(np.random.normal(0, 0.01, n)))
    
    # 生成成交量
    volume = np.random.randint(1000000, 10000000, n)
    
    # 构建DataFrame
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    # 生成交易信号（简单的均线策略）
    # 计算5日和20日移动平均线
    ma5 = df['close'].rolling(window=5).mean()
    ma20 = df['close'].rolling(window=20).mean()
    
    # 生成信号
    hold = pd.Series(0, index=dates)
    
    for i in range(1, len(dates)):
        if pd.notna(ma5.iloc[i]) and pd.notna(ma20.iloc[i]):
            # 金叉：5日均线上穿20日均线，买入
            if ma5.iloc[i-1] <= ma20.iloc[i-1] and ma5.iloc[i] > ma20.iloc[i]:
                hold.iloc[i] = 1
            # 死叉：5日均线下穿20日均线，卖出
            elif ma5.iloc[i-1] >= ma20.iloc[i-1] and ma5.iloc[i] < ma20.iloc[i]:
                hold.iloc[i] = -1
    
    print(f"生成了 {len(df)} 天的数据")
    print(f"生成了 {(hold==1).sum()} 个买入信号和 {(hold==-1).sum()} 个卖出信号")
    print("=" * 60 + "\n")
    
    return df, hold


def test_backtest_framework():
    """
    测试函数：使用随机生成的数据测试回测框架
    """
    print("\n" + "=" * 60)
    print("开始测试回测框架".center(56))
    print("=" * 60 + "\n")
    
    # 生成测试数据
    df, hold = generate_test_data(
        start_date='2020-01-01',
        end_date='2023-12-31',
        initial_price=100,
        volatility=0.02
    )
    
    # 创建回测对象
    backtest = BacktestFramework(
        df=df,
        hold=hold,
        initial_capital=100000,
        commission=0.0003,
        slippage=0.001
    )
    
    # 运行回测
    portfolio = backtest.run_backtest()
    
    # 计算并打印指标
    backtest.print_metrics()
    
    # 显示交易明细
    backtest.get_trade_details()
    
    # 绘制K线图
    print("绘制K线图...")
    backtest.plot_kline(figsize=(16, 10))
    
    # 绘制资产时序图
    print("绘制资产时序图...")
    backtest.plot_portfolio(figsize=(16, 10))
    
    print("\n" + "=" * 60)
    print("测试完成！".center(56))
    print("=" * 60)
    
    return backtest


# 主程序
if __name__ == "__main__":
    # 运行测试
    backtest = test_backtest_framework()
