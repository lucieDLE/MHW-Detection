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


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM cell and model adapted from 
    https://github.com/ndrplz/ConvLSTM_pytorch

    """
    def __init__(self, input_dims: int, hidden_dims: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.conv = nn.Conv2d(in_channels = input_dims + hidden_dims,
                              out_channels = 4 * hidden_dims,
                              kernel_size = kernel_size, 
                              padding=kernel_size // 2,
                              )

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        x = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = x.chunk(4, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMForecast(nn.Module):
    """ConvLSTM: spatial hidden state (B, hidden, H, W) unrolled over n_in timesteps.

    Input:  (B, n_in, H, W)
    Output: (B, n_out, H, W)
    """

    def __init__(self, n_in: int = 14, n_out: int = 1, hidden: int = 32):
        super().__init__()
        self.cell = ConvLSTMCell(input_dims=1, hidden_dims=hidden)
        self.head = nn.Conv2d(hidden, n_out, 1)
        self.skip = nn.Conv2d(1, n_out, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H, W = x.shape
        h = x.new_zeros(B, self.cell.hidden_dims, H, W)
        c = x.new_zeros(B, self.cell.hidden_dims, H, W)
        for t in range(T):
            h, c = self.cell(x[:, t:t+1], h, c)
        return self.head(h) + self.skip(x[:, -1:])


def build_model(name: str, n_in: int, n_out: int) -> nn.Module:
    if name == "lstm":
        return PixelLSTM(n_in=n_in, n_out=n_out)
    if name == "conv_lstm":
        return ConvLSTMForecast(n_in=n_in, n_out=n_out)
    raise ValueError(f"Unknown model '{name}'. Choose from: lstm, conv_lstm")
