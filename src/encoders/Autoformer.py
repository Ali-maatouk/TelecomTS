import torch
import torch.nn as nn
import torch.nn.functional as F
from encoders.utils.layers.normalization import NormalizationLayer
from encoders.utils.layers.Embed import DataEmbedding_wo_pos
from encoders.utils.layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from encoders.utils.layers.Autoformer_EncDec import (
    Encoder,
    EncoderLayer,
    my_Layernorm,
    series_decomp,
)


class Model(nn.Module):
    """
    Autoformer adapted for anomaly detection (sequence-level binary classification).
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # Decomposition
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
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

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.d_model)
        self.normalizer = NormalizationLayer(
            method="zscore", num_features=configs.enc_in
        )

    # Modify here to adapt for different tasks, and add anything needed to __init__.
    def forward_encoder(self, x_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        return enc_out

    def reconstruction_head(self, enc_out):
        dec_out = self.projection(enc_out)
        return dec_out

    def forward(self, x_enc):
        enc_out = self.forward_encoder(x_enc)  # enc_out: [B, L, D]
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # flatten (B, L*D)
        dec_out = self.projection(output)  # dec_out: [B, 1]
        return dec_out
