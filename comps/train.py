"""Train the comparables GNN on training sales, stopping when validation stops improving."""
from __future__ import annotations

import copy

import numpy as np
import torch

from comps.graph import CompsGraph
from comps.model import CompsGNN


def pick_device() -> torch.device:
    """A GPU if there is one (Colab, Kaggle), Apple's GPU on a Mac, otherwise the CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_tensors(x: np.ndarray, y: np.ndarray, g: CompsGraph, device="cpu") -> dict:
    t = {
        "x": torch.tensor(x), "y": torch.tensor(y, dtype=torch.float32),
        "nbr": torch.tensor(g.nbr), "mask": torch.tensor(g.mask),
        "dist_km": torch.tensor(g.dist_km), "days_ago": torch.tensor(g.days_ago),
    }
    return {k: v.to(device) for k, v in t.items()}


def train_gnn(x: np.ndarray, y_scaled: np.ndarray, g: CompsGraph, split: np.ndarray,
              epochs: int = 400, lr: float = 3e-3, hidden: int = 64, patience: int = 60,
              seed: int = 0, log=print) -> CompsGNN:
    """Full-batch training: every sale fits in memory at once, so no mini-batches are needed.

    The loss uses training sales only. Validation sales pick the stopping point. Test
    sales are never looked at here.
    """
    torch.manual_seed(seed)
    device = pick_device()
    log(f"training on {device}")
    t = to_tensors(x, y_scaled, g, device)
    train = torch.tensor(split == "train", device=device)
    val = torch.tensor(split == "val", device=device)
    model = CompsGNN(x.shape[1], hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best, best_state, since_best = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        pred, _ = model(**t)
        loss = torch.nn.functional.huber_loss(pred[train], t["y"][train], delta=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            pred, _ = model(**t)
            val_loss = torch.mean((pred[val] - t["y"][val]) ** 2).item()
        if val_loss < best - 1e-5:
            best, best_state, since_best = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            since_best += 1
        if epoch % 50 == 0:
            log(f"epoch {epoch:4d}  train loss {loss.item():.4f}  val mse {val_loss:.4f}")
        if since_best >= patience:
            log(f"stopped at epoch {epoch}: validation hadn't improved for {patience} epochs")
            break

    model.load_state_dict(best_state)
    return model.cpu()


@torch.no_grad()
def predict(model: CompsGNN, x: np.ndarray, y_scaled: np.ndarray, g: CompsGraph):
    model.eval()
    pred, weights = model(**to_tensors(x, y_scaled, g))
    return pred.numpy(), weights.numpy()
