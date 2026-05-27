import torch
import torch.nn as nn
import torch.nn.functional as F
from toto.model.backbone import TotoBackbone


class Model(nn.Module):
    """Toto backbone, randomly initialized for end-to-end training.

    Wraps Datadog's ``TotoBackbone`` (from the ``toto-ts`` package). The
    backbone applies a per-patch standardization scaler, a patch embedding,
    and a stack of transformer blocks that alternate between time-wise and
    space-wise (channel-wise) attention. We expose only the scaler +
    patch_embed + transformer path (no output head) since this repo trains a
    task-specific head separately.
    """

    _SCALER_CLS = "<class 'model.scaler.CausalPatchStdMeanScaler'>"
    _OUTPUT_DIST_CLS = "<class 'model.distribution.MixtureOfStudentTsOutput'>"

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        self.backbone = TotoBackbone(
            patch_size=configs.patch_size,
            stride=configs.stride,
            embed_dim=configs.embed_dim,
            num_layers=configs.e_layers,
            num_heads=configs.n_heads,
            mlp_hidden_dim=configs.d_ff,
            dropout=configs.dropout,
            spacewise_every_n_layers=configs.spacewise_every_n_layers,
            scaler_cls=self._SCALER_CLS,
            output_distribution_classes=[self._OUTPUT_DIST_CLS],
            output_distribution_kwargs={"k_components": configs.k_components},
            spacewise_first=False,
            use_memory_efficient_attention=False,
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.embed_dim, configs.d_model)

    def forward_encoder(self, x_enc):
        """x_enc: [B, T, C] -> [B, C, n_patches, embed_dim]"""
        x = x_enc.transpose(1, 2).contiguous()  # [B, C, T]
        B, C, T = x.shape
        padding = torch.zeros(B, C, T, dtype=torch.bool, device=x.device)
        id_mask = torch.zeros(B, C, T, dtype=torch.long, device=x.device)
        scaled, _, _ = self.backbone.scaler(
            x,
            weights=torch.ones_like(x),
            padding_mask=padding,
            prefix_length=None,
        )
        embeddings, reduced_id_mask = self.backbone.patch_embed(scaled, id_mask)
        return self.backbone.transformer(embeddings, reduced_id_mask, None)

    def forward(self, x_enc):
        h = self.forward_encoder(x_enc)  # [B, C, n_patches, embed_dim]
        h = h.mean(dim=(1, 2))            # [B, embed_dim]
        h = self.dropout(self.act(h))
        return self.projection(h)         # [B, d_model]
