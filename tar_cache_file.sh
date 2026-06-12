# Zarr = directories of many chunk files :bad for Git LFS (thousands of pointers, 
# HF file-count limits). Instead tar each store:


tar -cf data/deploy/initial_map.zarr.tar -C data/cache initial_map.zarr

tar -cf data/deploy/conv_lstm_n_in14_ACC_RMSE.zarr.tar -C data/cache conv_lstm_n_in14_ACC_RMSE.zarr

tar -cf data/deploy/conv_lstm_n_in14_forecast_fan.zarr.tar -C data/cache conv_lstm_n_in14_forecast_fan.zarr

tar -cf data/deploy/mhw.zarr.tar -C data/cache mhw.zarr

tar -cf data/deploy/ssta_high_res.zarr.tar -C data/cache_deploy ssta_high_res.zarr