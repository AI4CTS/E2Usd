import torch.nn as nn

import torch

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean




class DDEM(torch.nn.Module):
    def __init__(self, in_channels, channels, depth, reduced_size,
                 out_channels, kernel_size):
        super(DDEM, self).__init__()
        moving_ks = 5
        self.decompsition = series_decomp(moving_ks)

        self.trend_cnn = nn.Conv1d(in_channels, reduced_size,kernel_size=kernel_size)
        self.seasonal_cnn = nn.Conv1d(in_channels, reduced_size,kernel_size=kernel_size)

        self.trend_cnn.requires_grad_(False)
        self.seasonal_cnn.requires_grad_(False)
        self.trend_pooling = torch.nn.AdaptiveMaxPool1d(1)
        self.seasonal_pooling = torch.nn.AdaptiveMaxPool1d(1)

        self.linear_trend = torch.nn.Linear(reduced_size, out_channels)
        self.linear_seasonal = torch.nn.Linear(reduced_size, out_channels)

        self.linear = torch.nn.Linear(out_channels*2, out_channels)

        self.trade_off_freq = 33


    def forward(self, x):
        low_specx = torch.fft.rfft(x, dim=-1)
        low_specx = low_specx[:,:,:self.trade_off_freq]
        x = torch.fft.irfft(low_specx, dim=-1)*self.trade_off_freq/x.size(-1)
        seasonal_init, trend_init = self.decompsition(x)

        trend_x, seasonal_x = (self.trend_cnn(trend_init)), (self.seasonal_cnn(seasonal_init))
        trend_x_reduced, seasonal_x_reduced = self.trend_pooling(trend_x), self.seasonal_pooling(seasonal_x)

        trend_x_reduced, seasonal_x_reduced =trend_x_reduced.squeeze(2), seasonal_x_reduced.squeeze(2)

        trend_x_embedding, seasonal_x_embedding =  (self.linear_trend(trend_x_reduced)), (self.linear_seasonal(seasonal_x_reduced))

        embedding=torch.concat([trend_x_embedding,seasonal_x_embedding],dim=-1)
        embedding =self.linear(embedding)

        return embedding, trend_x_embedding, seasonal_x_embedding


