"""A one-layer graph attention network that values a home the way a valuer does.

For each home, every comparable sale sends one message: "I sold for this much; adjusted
for the differences between us, you're worth about this much." The home also makes its
own guess from its features alone. Attention then decides how much to trust each message,
and the estimate is the weighted average. The weights ARE the explanation: they say which
sales drove the price.

Prices are log prices, standardised with the training mean and spread, so "+0.1" means
roughly "10% of a typical spread dearer".
"""
from __future__ import annotations

import torch
from torch import nn


def mlp(n_in: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, 1))


class CompsGNN(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        edge_in = 2 * n_features + 2  # difference, the home itself, distance, days apart
        self.adjust = mlp(edge_in, hidden)       # how much dearer is this home than the comparable?
        self.score = mlp(edge_in, hidden)        # how much should we trust this comparable?
        self.own_estimate = mlp(n_features, hidden)  # a guess from the home's features alone
        self.own_score = mlp(n_features, hidden)     # how much to trust that guess

    def forward(self, x, y, nbr, mask, dist_km, days_ago):
        """
        x: (n, F) scaled features of every sale.  y: (n,) scaled log price of every sale.
        nbr, mask, dist_km, days_ago: (n, k) from the comparables graph.
        Returns the estimate for every sale (n,) and the attention weights (n, k + 1);
        column 0 is the home's own guess, columns 1..k are its comparables.

        A sale's own price y[i] is never used to estimate sale i: it only reaches later
        sales, through their comparables. tests/test_model.py checks this.
        """
        n, k = nbr.shape
        safe = nbr.clamp(min=0)  # padded slots point at sale 0, then get zero weight below
        xi = x.unsqueeze(1).expand(n, k, x.shape[1])
        xj = x[safe]
        edge = torch.cat([xi - xj, xi, dist_km.unsqueeze(-1), (days_ago / 365.0).unsqueeze(-1)], dim=-1)

        proposals = y[safe] + self.adjust(edge).squeeze(-1)                  # (n, k)
        scores = self.score(edge).squeeze(-1).masked_fill(~mask, float("-inf"))

        all_proposals = torch.cat([self.own_estimate(x), proposals], dim=1)   # (n, k + 1)
        all_scores = torch.cat([self.own_score(x), scores], dim=1)
        weights = torch.softmax(all_scores, dim=1)  # the own guess is always there, so never all -inf
        estimate = (weights * all_proposals).sum(dim=1)
        return estimate, weights
