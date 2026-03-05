import numpy as np
from sklearn.linear_model import LinearRegression

def linear_trend(X,y):
    reg = LinearRegression().fit(X, y)
    trend_decade = reg.coef_[0,0] * 10
    return trend_decade, reg

def rolling_mean(da, years, freq_per_year):
    da_resampled = da.resample(time='SME').mean('time')

    window = years * freq_per_year
    return da.rolling(time=window, center=True).mean()

def exceedance_fraction(da, q):
    threshold = da.quantile(q)
    exceed = da > threshold
    return exceed.groupby("time.year").mean("time")
