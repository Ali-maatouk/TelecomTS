import torch
import torch.nn as nn
import torch.nn.functional as F
from encoders.utils.layers.Embed import DataEmbedding
from encoders.utils.layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(
                configs.d_model, configs.d_ff, num_kernels=configs.num_kernels
            ),
            nn.GELU(),
            Inception_Block_V1(
                configs.d_ff, configs.d_model, num_kernels=configs.num_kernels
            ),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros(
                    [x.shape[0], length - (self.seq_len + self.pred_len), x.shape[2]]
                ).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            out = (
                out.reshape(B, length // period, period, N)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, : (self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        return res + x


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.in_chans = configs.enc_in

        self.model = nn.ModuleList(
            [TimesBlock(configs) for _ in range(configs.e_layers)]
        )

        self.enc_embedding = DataEmbedding(
            self.in_chans, configs.d_model, configs.embed, configs.freq, configs.dropout
        )

        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(configs.dropout)

        self.projection = nn.Linear(self.seq_len * configs.d_model, configs.d_model)

    def forward_encoder(self, x_enc):
        enc_out = self.enc_embedding(x_enc, None)  # (B,T,C)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        return enc_out

    def forward(self, x_enc):
        enc_out = self.forward_encoder(x_enc)  # (B,T,d_model)

        # activation + dropout
        output = self.act(enc_out)
        output = self.dropout(output)

        # flatten (B, T*d_model)
        output = output.reshape(output.shape[0], -1)

        # projection (B, d_model)
        output = self.projection(output)
        return output
