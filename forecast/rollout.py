import torch
import torch.nn as nn


@torch.no_grad()
def autoregressive_rollout(
    model: nn.Module,
    x_init: torch.Tensor,
    horizon: int,
    n_out: int = 1,
) -> torch.Tensor:
    """Roll a one-step (n_out=1) model to predict`horizon` days.
    Each step feeds the model's own prediction back as input.
    """
    model.eval()
    window = x_init.clone()
    preds = []
    steps = (horizon + n_out - 1) // n_out
    # print(window.shape)
    for _ in range(steps):
        yhat = model(window)
        preds.append(yhat)
        window = torch.cat([window[:, n_out:, :, :], yhat], dim=1)
    out = torch.cat(preds, dim=1)
    return out[:, :horizon, :, :]
