import torch
import torch.nn as nn
import copy


class NormalizationLayer(nn.Module):
    """
    Different normalization methods.
    """

    def __init__(self, method: str, num_features: int, san_cfg=None):
        super().__init__()
        self.method = method
        if method.lower() == "zscore":
            self.normalizer = ZScoreNormalizer()
        elif method.lower() == "minmax":
            self.normalizer = MinMaxNormalizer()
        elif method.lower() == "revin":
            self.normalizer = RevIN(num_features=num_features)
        elif method.lower() == "san":
            self.normalizer = Statistics_prediction(san_cfg)
        elif method.lower() in ["none", "identity"]:
            self.normalizer = nn.Identity()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def fit(self, x: torch.Tensor):
        """
        If the underlying normalizer supports fit(), delegate to it.
        """
        if hasattr(self.normalizer, "fit"):
            self.normalizer.fit(x)
        else:
            # e.g. Identity or MinMaxNormalizer without fit
            raise AttributeError(f"fit() not supported for method '{self.method}'")

    def forward(self, x, mode: str):
        """
        Applies normalization or denormalization.
        For nn.Identity, it simply returns the input tensor.

        :param x: Input tensor of shape [B, N, C, P], except for SAN
        :param mode: 'normalize' or 'denormalize'.
        """
        if self.method in ["none", "identity"]:
            return x
        return self.normalizer(x, mode=mode)


class Statistics_prediction(nn.Module):
    def __init__(self, configs):
        super(Statistics_prediction, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.period_len = configs.period_len
        self.channels = configs.enc_in if configs.features == "M" else 1
        self.station_type = configs.station_type

        self.seq_len_new = int(self.seq_len / self.period_len)
        self.pred_len_new = int(self.pred_len / self.period_len)
        self.epsilon = 1e-5
        self._build_model()
        self.weight = nn.Parameter(torch.ones(2, self.channels))

    def _build_model(self):
        args = copy.deepcopy(self.configs)
        args.seq_len = self.configs.seq_len // self.period_len
        args.label_len = self.configs.label_len // self.period_len
        args.enc_in = self.configs.enc_in
        args.dec_in = self.configs.dec_in
        args.moving_avg = 3
        args.c_out = self.configs.c_out
        args.pred_len = self.pred_len_new
        self.model = MLP(args, mode="mean").float()
        self.model_std = MLP(args, mode="std").float()

    def normalize(self, input):
        if self.station_type == "adaptive":
            bs, len, dim = input.shape
            input = input.reshape(bs, -1, self.period_len, dim)
            mean = torch.mean(input, dim=-2, keepdim=True)
            std = torch.std(input, dim=-2, keepdim=True)
            norm_input = (input - mean) / (std + self.epsilon)
            input = input.reshape(bs, len, dim)
            mean_all = torch.mean(input, dim=1, keepdim=True)
            outputs_mean = (
                self.model(mean.squeeze(2) - mean_all, input - mean_all)
                * self.weight[0]
                + mean_all * self.weight[1]
            )
            outputs_std = self.model_std(std.squeeze(2), input)
            outputs = torch.cat([outputs_mean, outputs_std], dim=-1)
            self.station_pred = outputs[:, -self.pred_len_new :, :]
            return norm_input.reshape(bs, len, dim)
        else:
            return input

    def de_normalize(self, input):
        if self.station_type == "adaptive":
            bs, len, dim = input.shape
            input = input.reshape(bs, -1, self.period_len, dim)
            mean = self.station_pred[:, :, : self.channels].unsqueeze(2)
            std = self.station_pred[:, :, self.channels :].unsqueeze(2)
            output = input * (std + self.epsilon) + mean
            return output.reshape(bs, len, dim)
        else:
            return input

    def forward(self, x, mode: str):
        if mode == "normalize":
            return self.normalize(x)
        elif mode == "denormalize":
            return self.de_normalize(x)


class MLP(nn.Module):
    def __init__(self, configs, mode):
        super(MLP, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.period_len = configs.period_len
        self.mode = mode
        if mode == "std":
            self.final_activation = nn.ReLU()
        else:
            self.final_activation = nn.Identity()
        self.input = nn.Linear(self.seq_len, 512)
        self.input_raw = nn.Linear(self.seq_len * self.period_len, 512)
        self.activation = nn.ReLU() if mode == "std" else nn.Tanh()
        self.output = nn.Linear(1024, self.pred_len)

    def forward(self, x, x_raw):
        x, x_raw = x.permute(0, 2, 1), x_raw.permute(0, 2, 1)
        x = self.input(x)
        x_raw = self.input_raw(x_raw)
        x = torch.cat([x, x_raw], dim=-1)
        x = self.output(self.activation(x))
        x = self.final_activation(x)
        return x.permute(0, 2, 1)


class RevIN(nn.Module):
    """Reversible Instance Normalization for Time Series"""

    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: The number of features or channels (C).
        :param eps: A value added for numerical stability.
        :param affine: If True, RevIN has learnable affine parameters.
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            # Reshape for broadcasting: [1, 1, C, 1]
            x = x * self.affine_weight.view(1, 1, self.num_features, 1)
            x = x + self.affine_bias.view(1, 1, self.num_features, 1)
        return x

    def _denormalize(self, x):
        if self.affine:
            # Reshape for broadcasting: [1, 1, C, 1]
            x = (x - self.affine_bias.view(1, 1, self.num_features, 1)) / (
                self.affine_weight.view(1, 1, self.num_features, 1) + self.eps**2
            )
        x = x * self.stdev + self.mean
        return x

    def forward(self, x, mode: str):
        if mode == "normalize":
            self._get_statistics(x)
            print("x shape:", x.shape)
            x = self._normalize(x)
        elif mode == "denormalize":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x


class ZScoreNormalizer(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, x: torch.Tensor):
        """
        x: (B, C, T)
        Compute and store mean/std over the last dim of x.
        """
        # detach so these stats aren’t part of autograd
        self.mean = x.mean(dim=-1, keepdim=True).detach()
        self.std = torch.sqrt(
            torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5
        ).detach()

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        mode == 'normalize':
          - if not yet fitted, compute stats via fit(x)
          - return (x - mean) / (std + eps)
        mode == 'denormalize':
          - return x * std + mean  (requires prior normalize/fit)
        """
        if mode == "normalize":
            if self.mean is None or self.std is None:
                self.fit(x)
            return (x - self.mean) / (self.std + self.eps)

        elif mode == "denormalize":
            if self.mean is None or self.std is None:
                raise RuntimeError("Must call normalize() (or fit) before denormalize.")
            return x * self.std + self.mean

        else:
            raise NotImplementedError(f"Mode '{mode}' is not implemented.")


class MinMaxNormalizer(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.min_val = None
        self.max_val = None

    def forward(self, x, mode: str):
        if mode == "normalize":
            self.min_val = torch.min(x, dim=-1, keepdim=True)[0].detach()
            self.max_val = torch.max(x, dim=-1, keepdim=True)[0].detach()
            return (x - self.min_val) / (self.max_val - self.min_val + self.eps)
        elif mode == "denormalize":
            return x * (self.max_val - self.min_val) + self.min_val
        else:
            raise NotImplementedError(f"Mode '{mode}' is not implemented.")
