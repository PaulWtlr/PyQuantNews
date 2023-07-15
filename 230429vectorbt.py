import numpy as np
import scipy.stats as stats
import vectorbt as vbt
# vectorbt is a package that combines backtesting and data science.
# It takes a “vectorized” approach to backtesting using pandas and NumPy. This means that instead of looping through every day of data, it operates on all the data at once.
# To avoid overfitting, pros use a technique called walk-forward optimization. Walk-forward optimization is a technique used to find the best settings for a trading strategy.


# Today, you’ll build a simple moving average crossover strategy. Then you’ll use walk-forward optimization to find the moving average windows that result in the best Sharpe ratio. 

# Then create an array of moving average windows to test and download price data.
windows = np.arange(10, 50)
price = vbt.YFData.download('AAPL').get('Close')

# Create the data splits for the walk-forward optimization.
(in_price, in_indexes), (out_price, out_indexes) = price.vbt.rolling_split(
    n=30, # This code segments the prices into 30 splits, each two years long, and reserves 180 days for the test.
    window_len=365 * 2,
    set_lens=(180,),
    left_to_right=False,
)

# Create the function that run the backtest
#This function builds two moving averages for each window you pass in.
def simulate_all_params(price, windows, **kwargs):
    fast_ma, slow_ma = vbt.MA.run_combs(
        price, windows, r=2, short_names=["fast", "slow"]
    )
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)

    pf = vbt.Portfolio.from_signals(price, entries, exits, **kwargs)
    return pf.sharpe_ratio()
#After the backtest is run, the function returns the Sharpe ratio.


### Next, you need to figure out the combination of windows that maximizes the Sharpe ratio.

# Returns the indexes in the DataFrame for the windows in each data split that maximizes the Sharpe ratio
def get_best_index(performance, higher_better=True): 
    if higher_better:
        return performance[performance.groupby('split_idx').idxmax()].index
    return performance[performance.groupby('split_idx').idxmin()].index

# Returns the window values
def get_best_params(best_index, level_name):
    return best_index.get_level_values(level_name).to_numpy()

# Runs the backtest with the windows that maximize the Sharpe ratio.
def simulate_best_params(price, best_fast_windows, best_slow_windows, **kwargs):
    
    fast_ma = vbt.MA.run(price, window=best_fast_windows, per_column=True)
    slow_ma = vbt.MA.run(price, window=best_slow_windows, per_column=True)

    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)

    pf = vbt.Portfolio.from_signals(price, entries, exits, **kwargs)
    return pf.sharpe_ratio()

### RUN THE ANALYSIS

# Start by optimizing the moving average windows on the in-sample data.
in_sharpe = simulate_all_params(
    in_price, 
    windows, 
    direction="both", 
    freq="d"
)

# Get the optimized windows and test them with out-of-sample data.
in_best_index = get_best_index(in_sharpe)

in_best_fast_windows = get_best_params(
    in_best_index,
    'fast_window'
)
in_best_slow_windows = get_best_params(
    in_best_index,
    'slow_window'
)
in_best_window_pairs = np.array(
    list(
        zip(
            in_best_fast_windows, 
            in_best_slow_windows
        )
    )
)

out_test_sharpe = simulate_best_params(
    out_price, 
    in_best_fast_windows, 
    in_best_slow_windows, 
    direction="both", 
    freq="d"
)

### Compare the results

in_sample_best = in_sharpe[in_best_index].values
out_sample_test = out_test_sharpe.values

t, p = stats.ttest_ind(
    a=out_sample_test,
    b=in_sample_best,
    alternative="greater"
)
print(t, p)