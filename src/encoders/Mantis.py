import torch
import torch.nn as nn
import torch.nn.functional as F
from mantis.architecture import MantisV1


class Model(nn.Module):
    """Mantis-V1 backbone, randomly initialized for end-to-end training.

    Wraps the per-channel MantisV1 encoder from the ``mantis-tsfm`` package.
    Each channel is processed independently as a univariate sequence; the
    CLS token from each channel is then mean-pooled across channels and
    projected to ``d_model`` for the downstream head.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        self.backbone = MantisV1(
            seq_len=configs.seq_len,
            hidden_dim=configs.hidden_dim,
            num_patches=configs.num_patches,
            transf_depth=configs.e_layers,
            transf_num_heads=configs.n_heads,
            transf_mlp_dim=configs.d_ff,
            transf_dim_head=configs.transf_dim_head,
            transf_dropout=configs.dropout,
            device="cpu",
            output_token="cls_token",
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.hidden_dim, configs.d_model)

    def forward_encoder(self, x_enc):
        """x_enc: [B, T, C] -> [B, C, hidden_dim]"""
        x = x_enc.transpose(1, 2).contiguous()  # [B, C, T]
        B, C, T = x.shape
        x = x.reshape(B * C, 1, T)
        if T != self.seq_len:
            x = F.interpolate(x, size=self.seq_len, mode="linear", align_corners=False)
        h = self.backbone(x)                     # [B*C, hidden_dim]
        return h.reshape(B, C, -1)               # [B, C, hidden_dim]

    def forward(self, x_enc):
        h = self.forward_encoder(x_enc)  # [B, C, hidden_dim]
        h = h.mean(dim=1)                 # [B, hidden_dim]
        h = self.dropout(self.act(h))
        return self.projection(h)         # [B, d_model]
