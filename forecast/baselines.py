import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader


def persistence_forecast(x: torch.Tensor, horizon: int) -> torch.Tensor:
    """Predict T_{t+k} = SSTA_t for all k. Often used as baseline to beat on short range.
    The idea behind is that temperature changes slowly so the today's temp/anomaly is likely to be the same tomorrow.
    Really strong for 1 to 7 days prediction.

    x: (B, n_in, H, W) — last frame is treated as 'today'
    returns: (B, horizon, H, W)
    """
    last = x[:, -1:, :, :]
    return last.expand(-1, horizon, -1, -1).clone()


class RidgeBaseline:
    """One pooled Ridge regression model per lead time.

    Treats every (time-step, ocean-pixel) pair as an independent sample with
    n_in values as features. Use sklearn Ridge Method.
    """

    def __init__(self, lead_times=(1, 3, 7, 14), alpha=1.0, max_fit_samples=300_000):
        self.lead_times = tuple(lead_times)
        self.alpha = alpha
        self.max_fit_samples = max_fit_samples
        self.models: dict = {}

    def fit(self, train_dataset, land_mask: np.ndarray) -> None:

        ocean = ~land_mask
        loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

        X_list: list = []
        Y_lists: dict = {k: [] for k in self.lead_times}
        n_collected = 0

        for x, y in loader:
            if n_collected >= self.max_fit_samples:
                break
            x_np = x.numpy()   # (B, n_in, H, W)
            y_np = y.numpy()   # (B, n_out, H, W)
            n_in = x_np.shape[1]

            x_ocean = x_np[:, :, ocean]                              # (B, n_in, n_ocean)
            X_batch = x_ocean.transpose(0, 2, 1).reshape(-1, n_in)  # (B*n_ocean, n_in)
            X_list.append(X_batch)

            for k in self.lead_times:
                if k - 1 < y_np.shape[1]:
                    Y_lists[k].append(y_np[:, k - 1, ocean].reshape(-1))

            n_collected += X_batch.shape[0]

        X = np.concatenate(X_list)
        for k in self.lead_times:
            Y = np.concatenate(Y_lists[k])
            self.models[k] = Ridge(alpha=self.alpha).fit(X, Y)
        print(f"  Ridge fitted on {len(X):,} (pixel×sample) pairs")

    def predict_all_leads(self, x: torch.Tensor, horizon: int) -> torch.Tensor:
        """x: (B, n_in, H, W) CPU tensor  →  (B, horizon, H, W) CPU tensor."""
        x_np = x.numpy()
        B, n_in, H, W = x_np.shape
        x_flat = x_np.transpose(0, 2, 3, 1).reshape(-1, n_in)   # (B*H*W, n_in)

        out = np.zeros((B, horizon, H, W), dtype=np.float32)
        for step in range(1, horizon + 1):
            if step in self.models:
                m = self.models[step]
            else:
                nearest = min(self.models.keys(), key=lambda k: abs(k - step))
                m = self.models[nearest]
            out[:, step - 1] = m.predict(x_flat).astype(np.float32).reshape(B, H, W)

        return torch.from_numpy(out)
