import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import yfinance as yf

# iShares 20+ Year Treasury Bond ETF (TLT)
tlt = yf.download("TLT", start="2002-01-01", end="2021-01-01")


#-----------------------------------------------------------------------------------
#--------------------------- PREPARE THE DATA --------------------------------------
#-----------------------------------------------------------------------------------

# We compute the log returns 
# "Adj Close" corresponds to the adjusted closing prices,
#  shift(1) method allow us to create a new Series where each value is shifted one position up. This effectively shifts the adjusted closing prices one day back, aligning each day's value with the previous day's closing price.
# Finally, the expression divides the original adjusted closing prices by the shifted adjusted closing prices. This calculation gives you the daily return of TLT as a ratio between the closing price of the current day and the closing price of the previous day.

tlt["log_return"] = np.log(tlt["Adj Close"] / tlt["Adj Close"].shift(1))

# We then add columns for the day and year
tlt["day_of_month"] = tlt.index.day
tlt["year"] = tlt.index.year


#-----------------------------------------------------------------------------------
#--------------------------- INVESTIGATE OUR HYPOTHESIS ----------------------------
#-----------------------------------------------------------------------------------

# We expect there to be positive returns in TLT toward the end of the month.
# We expect this because we think fund managers buy TLT at the end of the month.

# We expect there to be negative returns in TLT toward the beginning of the month. This is when fund managers sell their high-quality assets and go back to buying meme stocks.


grouped_by_day = tlt.groupby("day_of_month").log_return.mean()
grouped_by_day.plot.bar()
plt.show()

#-----------------------------------------------------------------------------------
#--------------------------- BUILD A SIMPLE TRADING STRATEGY -----------------------
#-----------------------------------------------------------------------------------

# Let’s build a naive strategy to test our hypothesis:
#   Buy and hold TLT during the last week of the month
#   Short and hold TLT during the first week of the month

tlt["first_week_returns"] = 0.0
tlt.loc[tlt.day_of_month <= 7, "first_week_returns"] = tlt[
    tlt.day_of_month <= 7
].log_return

tlt["last_week_returns"] = 0.0
tlt.loc[tlt.day_of_month >= 23, "last_week_returns"] = tlt[
    tlt.day_of_month >= 23
].log_return

tlt["last_week_less_first_week"] = tlt.last_week_returns - tlt.first_week_returns


#-----------------------------------------------------------------------------------
#--------------------------- PLOT RETURNS ------------------------------------------
#-----------------------------------------------------------------------------------

# Let’s create a naive backtest of our naive strategy to get a feel for the returns.
# The point of this is not to have a highly accurate, statistically significant backtest. It's to spend the shortest amount of time possible to see if this strategy is worth pursuing in more detail.
# First we’ll sum up the returns by year and plot them.

(
    tlt.groupby("year")
    .last_week_less_first_week.mean()
    .plot.bar()
)
plt.show()
# Let's take a look at the cumulative return by year

(
    tlt.groupby("year")
    .last_week_less_first_week.sum()
    .cumsum()
    .plot()
)
plt.show()
# We can do the same by day
tlt.last_week_less_first_week.cumsum().plot()
plt.show()


