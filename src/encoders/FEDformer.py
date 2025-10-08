import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.utils.layers.normalization import NormalizationLayer
from encoders.utils.layers.Embed import DataEmbedding
from encoders.utils.layers.AutoCorrelation import AutoCorrelationLayer
from encoders.utils.layers.FourierCorrelation import FourierBlock
from encoders.utils.layers.MultiWaveletCorrelation import MultiWaveletTransform

from encoders.utils.layers.Autoformer_EncDec import (
    Encoder,
    EncoderLayer,
    my_Layernorm,
    series_decomp,
)


class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieves O(N) complexity
    """

    def __init__(self, configs):
        """
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection methods, options: [random, low].
        modes: int, modes to be selected.
        """
        super().__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes

        # Decomposition
        self.decomp = series_decomp(configs.moving_avg)

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.dec_embedding = DataEmbedding(
            configs.dec_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        # Attention mechanism
        if self.version == "Wavelets":
            encoder_self_att = MultiWaveletTransform(
                ich=configs.d_model, L=1, base="legendre"
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=configs.d_model,
                out_channels=configs.d_model,
                n_heads=configs.n_heads,
                seq_len=self.seq_len,
                modes=self.modes,
                mode_select_method=self.mode_select,
            )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att, configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
        )

        # Loss
        self.loss_fn = nn.MSELoss()

        # Normalization
        self.normalizer = NormalizationLayer(
            method="zscore", num_features=configs.enc_in
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.d_model)

    def forward_encoder(self, x_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        return enc_out

    def forward(self, x_enc):
        """
        Args:
            x_enc: [B, C, T]  (channels-first input)
        Returns:
            loss: scalar MSE reconstruction loss
            recon_bct: [B, C, T] reconstructed sequence
        """
        enc_out = self.forward_encoder(x_enc)
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        dec_out = self.projection(output)
        return dec_out
