import numpy as np
import pandas as pd 
from config import *
from data import *
from utils import *


ds = load_dataset(DATA_PATH)
ds_ref = load_dataset(DATA_REF)

ds = remove_countries_data(ds, ds_ref)
da = ds.sst 

sst_med_sea = select_region(ds.sst, MED_BOUNDS)

print(sst_med_sea.shape, ds.sst.shape)

# 1) Display amomalie map
sst_gb = sst_med_sea.groupby("time.month", )
tos_clim = sst_gb.mean(dim='time')
tos_std = sst_gb.std(dim='time') # interannual variability for each month

# this would be the initial map displayed, allowing to click on region to display other metrics.
initial_map = tos_std.mean(dim='month')

# FIGURE 1
# plt.figure(figsize=(5,4))
# initial_map.plot()

tos_anom = sst_gb - tos_clim

# User click on one point on the map which selects coordinates. (to build)
sel_lat_idx, sel_lon_idx = 45, 35

# 2) Select specific point to analyze 
# anomalies: (seasonal cycle removed)
# Select point
tos_anom_selected = tos_anom.sel(lat=sel_lat_idx, lon=sel_lon_idx, method="nearest")

# 3) Trend and rolling average
time = da.time.dt.year + (da.time.dt.dayofyear - 1)/365.25
X = time.values.reshape(-1,1)
y = tos_anom_selected.values.reshape(-1,1)
print(X.shape, y.shape)

trend, reg = linear_trend(X,y)
rolled_avg = rolling_mean(tos_anom_selected, years=3, freq_per_year=FREQ_PER_YEAR_MIN)
y_month = np.arange(1990, 2023+1/FREQ_PER_YEAR_MIN, step=1/FREQ_PER_YEAR_MIN)


# FIGURE 2.1
# plt.figure(figsize=(20,10))
# plt.plot(X,y)
# plt.plot(X, reg.predict(X), linewidth=2)
# plt.plot(y_month,rolling_year_avg, 'r')

# plt.legend(["anomalies", f"Trend: {trend} ˚C/decade", f"rolling average {rolling_year_num} years"])
# plt.ylabel("Global mean tos anomaly")
# plt.xlabel("Time (months)")


q95 = np.quantile(tos_anom_selected, q=0.95)

# FIGURE 2.2
# plt.figure(figsize=(20,10))
# plt.plot(X,y, color='gray', alpha=0.6)

# plt.axhline(q95, color='red', linestyle='--', label='90th percentile')
# plt.scatter(X[y > q95],
#             y[y > q95],
#             color='red', s=20, label='Extreme')

# plt.legend(["anomalies","q95"])
# plt.ylabel("Global mean tos anomaly")
# plt.xlabel("Time (months)")


df = pd.DataFrame(data={'anomalies': y[:,0], 
                        'time': tos_anom_selected.time.dt.year},)

df = df.loc[df['time']!=2023] # only one month recorded
df['q95'] = (df['anomalies'] > q95).astype(int)

print(df['q95'].value_counts())

# FIGURE 2.3
# sns.set_theme(style="darkgrid")

# ax=sns.displot(data = df_sel, x = 'time', binwidth=1, kde=True, height=5)
# ax.set_xlabels('time (years)')
# ax.set_ylabels('number of extreme anomalie')
# plt.title('Number of Exteme Warm Anomalies per year')
print('all executed')