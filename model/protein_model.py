import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, TransformerConv


class ProtEncoderW(torch.nn.Module):
    def __init__(self, feature_dim, inter_dim, embd_dim, drop_p=0.2):
        super().__init__()
        self.gcn = GCNConv(feature_dim, inter_dim)
        self.act = nn.LeakyReLU(inplace=True)
        self.drop = nn.Dropout(drop_p)
        self.attn = TransformerConv(inter_dim, embd_dim, dropout=drop_p, edge_dim=1)

    def forward(self, x, edge_index, edge_weight):
        emb = self.gcn(x, edge_index, edge_weight)
        emb = self.act(emb)
        emb = self.drop(emb)
        edge_attr = edge_weight.view(-1, 1) if edge_weight is not None else None
        emb = self.attn(emb, edge_index, edge_attr)
        return emb


class ProtDecoderW(torch.nn.Module):
    def __init__(self, embd_dim, inter_dim, feature_dim, drop_p=0.2):
        super().__init__()
        self.attn_dec = TransformerConv(embd_dim, inter_dim, dropout=drop_p, edge_dim=1)
        self.act = nn.LeakyReLU(inplace=True)
        self.drop = nn.Dropout(drop_p)
        self.gcn_dec = GCNConv(inter_dim, feature_dim)

    def forward(self, z, edge_index, edge_weight=None):
        edge_attr = edge_weight.view(-1, 1) if edge_weight is not None else None
        dec = self.attn_dec(z, edge_index, edge_attr)
        dec = self.act(dec)
        dec = self.drop(dec)
        dec = self.gcn_dec(dec, edge_index, edge_weight)
        return dec


class SharedProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, h):
        return self.fc(h)


class DualAutoEncoder(nn.Module):
    def __init__(
        self,
        feature_dim1,
        feature_dim2,
        inter_dim,
        embd_dim,
        proj_out_dim,
        drop_p=0.2,
        merge_mode="concat",
        shared_decoder_input="shared",
    ):
        super().__init__()
        self.enc1 = ProtEncoderW(feature_dim1, inter_dim, embd_dim, drop_p)
        self.enc2 = ProtEncoderW(feature_dim2, inter_dim, embd_dim, drop_p)

        if merge_mode not in ["concat", "sum", "mean"]:
            raise ValueError("merge_mode must be concat, sum, or mean")
        self.merge_mode = merge_mode
        proj_in_dim = embd_dim * 2 if self.merge_mode == "concat" else embd_dim
        self.projector = SharedProjector(in_dim=proj_in_dim, out_dim=proj_out_dim)

        if shared_decoder_input not in ["shared", "separate"]:
            raise ValueError("shared_decoder_input must be shared or separate")
        self.shared_decoder_input = shared_decoder_input
        if shared_decoder_input == "separate":
            self.sep_head1 = nn.Identity()
            self.sep_head2 = nn.Identity()

        self.dec1 = ProtDecoderW(
            embd_dim=proj_out_dim,
            inter_dim=inter_dim,
            feature_dim=feature_dim1,
            drop_p=drop_p,
        )
        self.dec2 = ProtDecoderW(
            embd_dim=proj_out_dim,
            inter_dim=inter_dim,
            feature_dim=feature_dim2,
            drop_p=drop_p,
        )

    def forward(self, x1, x2, edge_index, edge_weight):
        z1 = self.enc1(x1, edge_index, edge_weight)
        z2 = self.enc2(x2, edge_index, edge_weight)

        if z1.size(0) != z2.size(0):
            raise ValueError("z1 and z2 must have the same number of protein nodes")

        if self.merge_mode == "concat":
            h = torch.cat([z1, z2], dim=-1)
        elif self.merge_mode == "sum":
            h = z1 + z2
        else:
            h = 0.5 * (z1 + z2)

        z_shared = self.projector(h)
        if self.shared_decoder_input == "shared":
            dec1 = self.dec1(z_shared, edge_index, edge_weight)
            dec2 = self.dec2(z_shared, edge_index, edge_weight)
        else:
            dec1 = self.dec1(self.sep_head1(z_shared), edge_index, edge_weight)
            dec2 = self.dec2(self.sep_head2(z_shared), edge_index, edge_weight)

        return z1, z2, z_shared, dec1, dec2
