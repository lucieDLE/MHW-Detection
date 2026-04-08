import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import seaborn as sns
import pandas as pd 

import imageio.v2 as imageio
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


ds_ssta = xr.open_dataset('../data/cache/ssta_high_res_2.nc')

ssta = ds_ssta.sst
x = ssta.values 
x = x[~np.isnan(x)]

p099, p001 = np.quantile(x,q=0.99), np.quantile(x,q=0.01)
print(p001, p099)

OUT_PATH = Path('../assets/sst_weekly_all.mp4')
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

FRAME_STEP = 2  # increase to skip frames, e.g. 2 or 4

writer = imageio.get_writer(str(OUT_PATH), fps=12, codec='libx264', quality=8)

fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
im = ax.imshow(ssta.isel(time=0).values, cmap='RdBu_r', vmin=p001, vmax=p099, origin='lower')

ax.set_axis_off()
title = ax.set_title(str(ssta.time.values[0])[:10])

print(ssta.sizes['time'])
# for i in range(0, 100, FRAME_STEP):
for i in range(0, ssta.sizes['time'], FRAME_STEP):

    print(f"{i} / {ssta.sizes['time']}")

    frame = ssta.isel(time=i).values
    ax.imshow(frame, cmap='RdBu_r', vmin=p001, vmax=p099, origin='lower')
    ax.set_axis_off()
    title = ax.set_title(str(ssta.isel(time=i).time.values)[:10])

    fig.canvas.draw()
    img_array = np.array(fig.canvas.renderer.buffer_rgba())

    writer.append_data(img_array)

writer.close()
plt.close(fig)
print(f'Saved video to {OUT_PATH}')
