import torch.nn as nn


def _to_cf(x):
    return x.transpose(1, 2)


def _to_cl(x):
    return x.transpose(1, 2)


class CellEncoderC(nn.Module):
    """Conv1d and self-attention encoder for cell or spatial-location profiles."""

    def __init__(
        self,
        feature_dim,
        inter_dim,
        embd_dim,
        drop_p=0.2,
        kernel_size=3,
        num_heads=4,
        attn_dropout=0.0,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(feature_dim, embd_dim, kernel_size, padding=pad)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.drop1 = nn.Dropout(drop_p)
        self.attn = nn.MultiheadAttention(
            embed_dim=embd_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.ln_attn = nn.LayerNorm(embd_dim)

    def forward(self, x, edge_index=None, edge_weight=None):
        x_cf = _to_cf(x)
        h = self.conv1(x_cf)
        h = self.act1(h)
        h = self.drop1(h)
        h = _to_cl(h)
        attn_out, _ = self.attn(h, h, h)
        return self.ln_attn(attn_out + h)


class CellDecoderC(nn.Module):
    """Self-attention and Conv1d decoder for expression-profile reconstruction."""

    def __init__(
        self,
        embd_dim,
        inter_dim,
        feature_dim,
        drop_p=0.2,
        kernel_size=3,
        num_heads=4,
        attn_dropout=0.0,
    ):
        super().__init__()
        pad = kernel_size // 2
        self.attn = nn.MultiheadAttention(
            embed_dim=embd_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.ln_attn = nn.LayerNorm(embd_dim)
        self.drop1 = nn.Dropout(drop_p)
        self.conv_out = nn.Conv1d(embd_dim, feature_dim, kernel_size, padding=pad)
        self.bn_out = nn.BatchNorm1d(feature_dim)

    def forward(self, z, edge_index=None, edge_weight=None):
        attn_out, _ = self.attn(z, z, z)
        h = self.ln_attn(attn_out + z)
        h = self.drop1(h)
        x_rec = self.conv_out(_to_cf(h))
        x_rec = self.bn_out(x_rec)
        return _to_cl(x_rec)


class CellAE(nn.Module):
    def __init__(
        self,
        feature_dim,
        inter_dim,
        embd_dim,
        drop_p=0.2,
        kernel_size=3,
        num_heads=4,
        attn_dropout=0.0,
    ):
        super().__init__()
        self.encoder = CellEncoderC(
            feature_dim,
            inter_dim,
            embd_dim,
            drop_p=drop_p,
            kernel_size=kernel_size,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
        )
        self.decoder = CellDecoderC(
            embd_dim,
            inter_dim,
            feature_dim,
            drop_p=drop_p,
            kernel_size=kernel_size,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec
