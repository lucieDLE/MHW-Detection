import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelLSTM(nn.Module):
    """Per-pixel LSTM baseline: applied independently to every grid cell.

    Input :  (B, n_in,  H, W)  — reshaped to (B*H*W, n_in, 1)
    Output:  (B, n_out, H, W)
    """

    def __init__(self, n_in: int = 14, n_out: int = 1, hidden: int = 32, num_layers: int = 1):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(B * H * W, T, 1)
        out, _ = self.lstm(seq)
        y = self.head(out[:, -1, :]) # (B*H*W, n_out)
        return y.reshape(B, H, W, self.n_out).permute(0, 3, 1, 2).contiguous()
