import argparse
import os
import random
import sys


def _configure_cuda_from_cli():
    if "--cuda_visible_devices" in sys.argv:
        index = sys.argv.index("--cuda_visible_devices")
        if index + 1 < len(sys.argv):
            os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[index + 1]


_configure_cuda_from_cli()
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from protein_model import DualAutoEncoder
from cell_model_1 import CellAE


def set_seed(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


class KMeansLoss(nn.Module):
    def __init__(self, intra_weight=1.0, inter_weight=1.0, margin=1.0, reverse=False, normalize=False):
        super().__init__()
        self.intra_weight = intra_weight
        self.inter_weight = inter_weight
        self.margin = margin
        self.reverse = reverse
        self.normalize = normalize

    @staticmethod
    def _pairwise_sq_dists(x):
        x2 = (x * x).sum(dim=1, keepdim=True)
        return (x2 + x2.T - 2 * (x @ x.T)).clamp_min(0.0)

    def forward(self, emb, labels):
        if self.normalize:
            emb = F.normalize(emb, dim=1)
        classes = labels.unique(sorted=True)
        centroids = torch.stack([emb[labels == c].mean(dim=0) for c in classes], dim=0)
        idx = torch.searchsorted(classes, labels)
        assigned = centroids[idx]
        intra = (emb - assigned).pow(2).sum(dim=1).mean()
        if self.inter_weight > 0 and centroids.size(0) > 1:
            d2 = self._pairwise_sq_dists(centroids)
            triu = torch.triu_indices(d2.size(0), d2.size(1), offset=1, device=emb.device)
            pair_d2 = d2[triu[0], triu[1]]
            inter = F.relu(self.margin * self.margin - pair_d2).mean()
        else:
            inter = emb.new_tensor(0.0)
        loss = self.intra_weight * intra + self.inter_weight * inter
        return -loss if self.reverse else loss


def pairwise_euclid_dists(x, eps=1e-8):
    x2 = (x * x).sum(dim=1, keepdim=True)
    sq_d = (x2 + x2.T - 2.0 * (x @ x.T)).clamp_min(0.0)
    return torch.sqrt(sq_d + eps)


def euclid_structure_loss_by_batch(x_all, z_emb, batch_labels, eps=1e-8):
    x = x_all.squeeze(1) if x_all.dim() == 3 else x_all
    classes = batch_labels.unique(sorted=True)
    losses = []
    for c in classes:
        mask = batch_labels == c
        n_b = int(mask.sum().item())
        if n_b < 2:
            continue
        dx = pairwise_euclid_dists(x[mask], eps=eps)
        dz = pairwise_euclid_dists(z_emb[mask], eps=eps)
        tri = torch.triu_indices(n_b, n_b, offset=1, device=dx.device)
        losses.append((dz[tri[0], tri[1]] - dx[tri[0], tri[1]]).pow(2).mean())
    return torch.stack(losses).mean() if losses else z_emb.new_tensor(0.0)


class PPILoss(nn.Module):
    def __init__(self, mode="dot_sigmoid", reduction="mean", eps=1e-8):
        super().__init__()
        self.mode = mode
        self.eps = eps
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(self, emb, edge_index, target_w):
        src = emb[edge_index[0]]
        dst = emb[edge_index[1]]
        if self.mode == "cosine":
            pred = (F.normalize(src, dim=1, eps=self.eps) * F.normalize(dst, dim=1, eps=self.eps)).sum(dim=1)
        elif self.mode == "dot":
            pred = (src * dst).sum(dim=1)
        elif self.mode == "dot_sigmoid":
            pred = torch.sigmoid((src * dst).sum(dim=1))
        else:
            raise ValueError(f"Unknown PPI similarity mode: {self.mode}")
        return self.mse(pred, target_w.view(-1))


def masked_mse(pred, target, mask):
    diff = (pred - target)[mask]
    return diff.pow(2).mean() if diff.numel() else pred.new_tensor(0.0)


class ModelTrain(nn.Module):
    def __init__(self, x_1, x_2, x_all, edge_index, edge_weight, cell_type, cell_batch,
                 inter_dim1, inter_dim2, embd_dim1, embd_dim2, proj_out_dim, device,
                 epochs, save_dir, merge_mode="concat", shared_decoder_input="shared",
                 kernel_size=3, seed=42, lr=1e-3, wd=1e-4, drop_p=0.2, rec_loss_w=1.0,
                 kmeans_loss_w=1.0, kmeans_intra_w=1.0, kmeans_inter_w=1.0,
                 kmeans_margin=1.0, normalize_emb=False, reverse_batch=False,
                 corr_struct_w=0.01, corr_eps=1e-8, ppi_loss_w=10.0,
                 ppi_sim="dot_sigmoid"):
        super().__init__()
        self.x_1 = x_1.to(device)
        self.x_2 = x_2.to(device)
        self.x_all = x_all.to(device)
        self.edge_index = edge_index.to(device) if edge_index is not None else None
        self.edge_weight = edge_weight.to(device) if edge_weight is not None else None
        self.cell_type = cell_type.to(device) if cell_type is not None else None
        self.cell_batch = cell_batch.to(device) if cell_batch is not None else None
        self.device = device
        self.epochs = epochs
        self.save_dir = save_dir
        self.rec_loss_w = rec_loss_w
        self.kmeans_loss_w = kmeans_loss_w
        self.corr_struct_w = corr_struct_w
        self.corr_eps = corr_eps
        self.ppi_loss_w = ppi_loss_w
        self.ppi_sim = ppi_sim
        os.makedirs(save_dir, exist_ok=True)

        self.pmodel = DualAutoEncoder(
            x_1.shape[1], x_2.shape[1], inter_dim1, embd_dim1, proj_out_dim,
            drop_p=drop_p, merge_mode=merge_mode, shared_decoder_input=shared_decoder_input,
        ).to(device)
        self.cmodel = CellAE(
            x_all.shape[-1], inter_dim2, embd_dim2, drop_p=drop_p, kernel_size=kernel_size,
        ).to(device)
        self.optimizer1 = torch.optim.AdamW(self.pmodel.parameters(), lr=lr, weight_decay=wd)
        self.optimizer2 = torch.optim.AdamW(self.cmodel.parameters(), lr=lr, weight_decay=wd)
        self.scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer1, "min", factor=0.5, patience=5)
        self.scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer2, "min", factor=0.5, patience=5)
        self.kmeans_loss_batch = KMeansLoss(
            intra_weight=kmeans_intra_w, inter_weight=kmeans_inter_w,
            margin=kmeans_margin, normalize=normalize_emb, reverse=reverse_batch,
        )
        self.mask_x1 = x_1 != 0
        self.mask_x2 = x_2 != 0
        self.mask_xall = x_all != 0
        self.ppi_loss_fn = PPILoss(mode=ppi_sim)

    def train_model(self):
        print("Start training...")
        loss_log = []
        last_pro_embd = None
        last_cell_embd = None
        for ep in range(1, self.epochs + 1):
            self.pmodel.train()
            self.cmodel.train()
            self.optimizer1.zero_grad(set_to_none=True)
            self.optimizer2.zero_grad(set_to_none=True)
            _, _, pro_embd, pro_dec1, pro_dec2 = self.pmodel(
                self.x_1, self.x_2, self.edge_index, self.edge_weight,
            )
            cell_embd, cell_dec = self.cmodel(self.x_all)
            cell_embd = cell_embd.squeeze(1)
            corr_struct_loss = euclid_structure_loss_by_batch(
                self.x_all, cell_embd, self.cell_batch, eps=self.corr_eps,
            )
            km_batch = self.kmeans_loss_batch(cell_embd, self.cell_batch)
            rec_p1 = masked_mse(pro_dec1, self.x_1, self.mask_x1)
            rec_p2 = masked_mse(pro_dec2, self.x_2, self.mask_x2)
            rec_c = masked_mse(cell_dec, self.x_all, self.mask_xall)
            rec_a_input = (cell_embd @ pro_embd.T).unsqueeze(1)
            rec_a = masked_mse(rec_a_input, self.x_all, self.mask_xall)
            if self.edge_index is not None and self.edge_weight is not None and self.ppi_loss_w != 0.0:
                ppi_loss = self.ppi_loss_fn(pro_embd, self.edge_index, self.edge_weight)
            else:
                ppi_loss = self.x_all.new_tensor(0.0)
            total = self.rec_loss_w * (rec_p1 + rec_p2 + rec_c + rec_a) \
                    - self.kmeans_loss_w * km_batch \
                    + self.corr_struct_w * corr_struct_loss \
                    + self.ppi_loss_w * ppi_loss
            total.backward()
            self.optimizer1.step()
            self.optimizer2.step()
            self.scheduler1.step(total.detach())
            self.scheduler2.step(total.detach())
            last_pro_embd = pro_embd.detach()
            last_cell_embd = cell_embd.detach()
            loss_log.append({
                "epoch": ep,
                "total": float(total.item()),
                "corr_struct": float(corr_struct_loss.item()),
                "kmeans_batch": float(km_batch.item()),
                "recon_pro1": float(rec_p1.item()),
                "recon_pro2": float(rec_p2.item()),
                "recon_cell": float(rec_c.item()),
                "recon_all": float(rec_a.item()),
                "ppi_loss": float(ppi_loss.item()),
            })
            if ep % 10 == 0 or ep == 1 or ep == self.epochs:
                print(
                    f"Ep {ep:04d} | Total {total.item():.4f} | "
                    f"CorrStruct {corr_struct_loss.item():.4f} | KM(batch) {km_batch.item():.4f} | "
                    f"PPI {ppi_loss.item():.4f} | RecP1 {rec_p1.item():.4f} | "
                    f"RecP2 {rec_p2.item():.4f} | RecC {rec_c.item():.4f} | RecA {rec_a.item():.4f}"
                )

        torch.save(self.pmodel.state_dict(), os.path.join(self.save_dir, "final_pmodel.pth"))
        torch.save(self.cmodel.state_dict(), os.path.join(self.save_dir, "final_cmodel.pth"))
        np.save(os.path.join(self.save_dir, "final_protein_emb.npy"), last_pro_embd.cpu().numpy())
        np.save(os.path.join(self.save_dir, "final_cell_emb.npy"), last_cell_embd.cpu().numpy())
        pd.DataFrame(loss_log).to_excel(
            os.path.join(self.save_dir, "training_losses.xlsx"), sheet_name="losses", index=False,
        )
        return self.pmodel, self.cmodel, last_pro_embd, last_cell_embd, loss_log


def parse_args():
    parser = argparse.ArgumentParser(description="Run scPPInt for one single-cell protein integration dataset.")
    parser.add_argument("--merged_data", required=True, help="Merged protein expression CSV; proteins in rows and cells in columns.")
    parser.add_argument("--x1_data", required=True, help="First input protein expression CSV.")
    parser.add_argument("--x2_data", required=True, help="Second input protein expression CSV.")
    parser.add_argument("--x1_cell", required=True, help="Cell annotation CSV for x1_data.")
    parser.add_argument("--x2_cell", required=True, help="Cell annotation CSV for x2_data.")
    parser.add_argument("--ppi", required=True, help="PPI edge CSV with protein1, protein2, and score columns.")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--inter_dim1", type=int, default=128)
    parser.add_argument("--inter_dim2", type=int, default=512)
    parser.add_argument("--embd_dim1", type=int, default=128)
    parser.add_argument("--embd_dim2", type=int, default=128)
    parser.add_argument("--proj_out_dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--drop_p", type=float, default=0.2)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--rec_loss_w", type=float, default=1.0)
    parser.add_argument("--kmeans_loss_w", type=float, default=1.0)
    parser.add_argument("--kmeans_intra_w", type=float, default=1.0)
    parser.add_argument("--kmeans_inter_w", type=float, default=1.0)
    parser.add_argument("--kmeans_margin", type=float, default=1.0)
    parser.add_argument("--corr_struct_w", type=float, default=0.01)
    parser.add_argument("--corr_eps", type=float, default=1e-8)
    parser.add_argument("--ppi_loss_w", type=float, default=10.0)
    parser.add_argument("--merge_mode", choices=["concat", "sum", "mean"], default="concat")
    parser.add_argument("--shared_decoder_input", choices=["shared", "separate"], default="shared")
    parser.add_argument("--ppi_sim", choices=["dot_sigmoid", "cosine", "dot"], default="dot_sigmoid")
    parser.add_argument("--device", default=None, help="Device such as cpu or cuda; default is auto-detection.")
    parser.add_argument("--cuda_visible_devices", default=None, help="CUDA_VISIBLE_DEVICES value, for example 0.")
    parser.add_argument("--normalize_emb", action="store_true")
    parser.add_argument("--reverse_batch", action="store_true")
    return parser.parse_args()


def load_expression(path):
    return pd.read_csv(path, index_col=0)


def main():
    args = parse_args()
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    set_seed(args.seed, deterministic=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("You will use : ", device)

    merged = load_expression(args.merged_data)
    x1_df = load_expression(args.x1_data)
    x2_df = load_expression(args.x2_data)
    cell1 = pd.read_csv(args.x1_cell, index_col=0)
    cell2 = pd.read_csv(args.x2_cell, index_col=0)
    ppi = pd.read_csv(args.ppi)

    merged.columns = merged.columns.astype(str)
    x1_df.columns = x1_df.columns.astype(str)
    x2_df.columns = x2_df.columns.astype(str)
    cell = pd.concat([cell1, cell2], axis=0)
    cell.index = cell.index.astype(str)
    missing_cells = merged.columns.difference(cell.index)
    if len(missing_cells):
        raise ValueError(f"Cell annotations are missing for {len(missing_cells)} merged_data columns")
    cell = cell.loc[merged.columns]
    for required_col in ["cell_type", "cell_batch"]:
        if required_col not in cell.columns:
            raise ValueError(f"Cell annotation files must contain '{required_col}'")

    protein_ids = merged.index.astype(str)
    id_to_idx = {pid: i for i, pid in enumerate(protein_ids)}
    ppi = ppi[ppi["protein1"].astype(str).isin(id_to_idx) & ppi["protein2"].astype(str).isin(id_to_idx)].copy()
    edge_index_np = ppi[["protein1", "protein2"]].astype(str).replace(id_to_idx).to_numpy().T
    edge_weight_np = ppi["score"].to_numpy(dtype=np.float32)

    x1_columns = [column for column in x1_df.columns if column in merged.columns]
    x2_columns = [column for column in x2_df.columns if column in merged.columns]
    if len(x1_columns) == 0 or len(x2_columns) == 0:
        raise ValueError("x1_data and x2_data must contain cells also present in merged_data")

    # Preserve the original data1 behavior: use x1/x2 columns to select values from merged_data.
    x1 = torch.from_numpy(merged.loc[:, x1_columns].to_numpy(dtype=np.float32)).to(device)
    x2 = torch.from_numpy(merged.loc[:, x2_columns].to_numpy(dtype=np.float32)).to(device)
    x_all = torch.from_numpy(merged.to_numpy(dtype=np.float32).T).unsqueeze(1).to(device)
    edge_index = torch.from_numpy(edge_index_np).long().to(device)
    edge_weight = torch.from_numpy(edge_weight_np).float().to(device)

    le_type = LabelEncoder()
    le_batch = LabelEncoder()
    cell_type_idx = torch.tensor(le_type.fit_transform(cell["cell_type"].astype(str)), dtype=torch.long, device=device)
    cell_batch_idx = torch.tensor(le_batch.fit_transform(cell["cell_batch"].astype(str)), dtype=torch.long, device=device)

    model = ModelTrain(
        x1, x2, x_all, edge_index, edge_weight, cell_type_idx, cell_batch_idx,
        args.inter_dim1, args.inter_dim2, args.embd_dim1, args.embd_dim2, args.proj_out_dim,
        device, args.epochs, args.output_dir, merge_mode=args.merge_mode,
        shared_decoder_input=args.shared_decoder_input, kernel_size=args.kernel_size,
        seed=args.seed, lr=args.lr, wd=args.wd, drop_p=args.drop_p,
        rec_loss_w=args.rec_loss_w, kmeans_loss_w=args.kmeans_loss_w,
        kmeans_intra_w=args.kmeans_intra_w, kmeans_inter_w=args.kmeans_inter_w,
        kmeans_margin=args.kmeans_margin, normalize_emb=args.normalize_emb,
        reverse_batch=args.reverse_batch, corr_struct_w=args.corr_struct_w,
        corr_eps=args.corr_eps, ppi_loss_w=args.ppi_loss_w, ppi_sim=args.ppi_sim,
    )
    _, _, pro_embd, cell_embd, _ = model.train_model()

    integrated = pd.DataFrame(
        (cell_embd @ pro_embd.T).detach().cpu().numpy().T,
        index=merged.index,
        columns=merged.columns,
    )
    integrated.index.name = merged.index.name
    integrated.to_csv(os.path.join(args.output_dir, "integrated_protein_expression.csv"))
    print("Saved integrated protein expression to:", os.path.join(args.output_dir, "integrated_protein_expression.csv"))


if __name__ == "__main__":
    main()
