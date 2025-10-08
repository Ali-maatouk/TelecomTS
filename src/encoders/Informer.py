import torch
import torch.nn as nn
import torch.nn.functional as F
from encoders.utils.layers.normalization import NormalizationLayer
from encoders.utils.layers.Transformer_EncDec import Encoder, EncoderLayer
from encoders.utils.layers.SelfAttention_Family import ProbAttention, AttentionLayer
from encoders.utils.layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    Informer with ProbSparse attention in O(LlogL) complexity
    """

    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len

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

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(
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
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            None,
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        self.normalizer = NormalizationLayer(
            method="zscore", num_features=configs.enc_in
        )
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.d_model)

    def forward_encoder(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        return enc_out

    def reconstruction_head(self, enc_out):
        dec_out = self.projection(enc_out)
        return dec_out

    def forward_loss(self, pred, target):
        return self.loss_fn(pred, target)

    def forward(self, x_enc):
        x_enc_out = self.forward_encoder(x_enc)
        output = self.act(
            x_enc_out
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(
            output.shape[0], -1
        )  # (batch_size, seq_length * d_model)
        output = self.projection(output)
        return output
