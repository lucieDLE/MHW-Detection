import numpy as np
import torch


def persistence_forecast(x: torch.Tensor, horizon: int) -> torch.Tensor:
    """Predict T_{t+k} = SSTA_t for all k. Often used as baseline to beat on short range.
    The idea behind is that temperature changes slowly so the today's temp/anomaly is likely to be the same tomorrow.
    Really strong for 1 to 7 days prediction.

    x: (B, n_in, H, W) — last frame is treated as 'today'
    returns: (B, horizon, H, W)
    """
    last = x[:, -1:, :, :]
    return last.expand(-1, horizon, -1, -1).clone()
