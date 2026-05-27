import torch
import torch.nn as nn
import torch.nn.functional as F
from chronos import ChronosPipeline
from transformers import T5Config, T5EncoderModel


class Model(nn.Module):
    """Chronos-style encoder, randomly initialized for end-to-end training.

    Reuses the ``MeanScaleUniformBins`` tokenizer from ``amazon/chronos-t5-tiny``
    (a small public checkpoint downloaded once and cached) to discretize each
    channel into tokens, then feeds them through a T5 encoder whose weights
    are initialized from scratch with the hyperparameters given in
    ``configs.{d_model, d_kv, d_ff, n_heads, e_layers, dropout}``.

    The pretrained encoder weights are discarded immediately after the
    tokenizer is extracted, so only the tokenizer's binning parameters are
    inherited from Chronos.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-tiny")
        self.tokenizer = pipe.tokenizer
        vocab_size = pipe.model.model.config.vocab_size
        del pipe

        t5_cfg = T5Config(
            vocab_size=vocab_size,
            d_model=configs.d_model,
            d_kv=configs.d_kv,
            d_ff=configs.d_ff,
            num_heads=configs.n_heads,
            num_layers=configs.e_layers,
            feed_forward_proj="gated-gelu",
            dropout_rate=configs.dropout,
        )
        self.encoder = T5EncoderModel(t5_cfg).encoder

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.d_model, configs.d_model)

    def forward_encoder(self, x_enc):
        """x_enc: [B, T, C] -> [B, C, T, d_model]"""
        B, T, C = x_enc.shape
        x_flat = x_enc.transpose(1, 2).reshape(B * C, T).detach().cpu()
        ids, mask, _ = self.tokenizer.context_input_transform(x_flat)
        ids = ids[:, :-1].to(x_enc.device)
        mask = mask[:, :-1].to(x_enc.device)
        out = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        return out.reshape(B, C, T, self.d_model)

    def forward(self, x_enc):
        h = self.forward_encoder(x_enc)
        h = h.mean(dim=(1, 2))
        h = self.dropout(self.act(h))
        return self.projection(h)
