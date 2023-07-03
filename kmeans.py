# Import the modules
from math import sqrt

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
plt.rc("font", size=10)

from openbb_terminal.sdk import openbb

#Set up the data (Dow Jones stocks)
dji = (
    pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')[1]
)
symbols = dji.Symbol.tolist()
data = openbb.stocks.ca.hist(
    symbols, 
    start_date="2020-01-01",
    end_date="2022-12-31"
)

#Data preprocessing

moments = (
    data
    .pct_change()
    .describe()
    .T[["mean", "std"]]
    .rename(columns={"mean": "returns", "std": "vol"})
) * [252, sqrt(252)]

#Do KMeans Clustering

sse = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(moments)
    sse.append(kmeans.inertia_)

plt.plot(range(2, 15), sse)
plt.title("Elbow Curve");

# Build and plot the clusters 

kmeans = KMeans(n_clusters=5, n_init=10).fit(moments)
plt.scatter(
    moments.returns, 
    moments.vol, 
    c=kmeans.labels_, 
    cmap="rainbow",
);

plt.title("Dow Jones stocks by return and volatility (K=5)")
for i in range(len(moments.index)):
    txt = f"{moments.index[i]} ({kmeans.labels_[i]})"
    xy = tuple(moments.iloc[i, :] + [0, 0.01])
    plt.annotate(txt, xy)