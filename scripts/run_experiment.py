"""Run the whole experiment: data -> graph -> baselines -> GNN -> results and demo files.

    python scripts/run_experiment.py                 # simulated city (default)
    python scripts/run_experiment.py --csv sales.csv # your own sales, same columns
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comps.data import load_sales, simulate_sales, time_split  # noqa: E402
from comps.evaluate import (  # noqa: E402
    comps_features, comps_median_baseline, conformal_halfwidth, fit_lightgbm, summarise,
)
from comps.features import Scaler, raw_features  # noqa: E402
from comps.graph import build_comps_graph  # noqa: E402
from comps.train import predict, train_gnn  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="real sales CSV; default is the simulated city")
    ap.add_argument("--n", type=int, default=15000, help="simulated sales")
    ap.add_argument("--k", type=int, default=10, help="comparables per sale")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    started = time.time()

    df = load_sales(args.csv) if args.csv else simulate_sales(args.n)
    source = args.csv or f"SIMULATED city, {args.n} sales"
    split = time_split(df)
    log_price = np.log(df["price"].to_numpy())
    is_train, is_val, is_test = split == "train", split == "val", split == "test"
    print(f"{len(df)} sales from {source}: {is_train.sum()} train, {is_val.sum()} val, {is_test.sum()} test")

    g = build_comps_graph(df, k=args.k)
    print(f"comparables graph: {g.mask.sum()} edges, {np.mean(g.mask.any(1)) * 100:.1f}% of sales have at least one")

    raw = raw_features(df)
    x = Scaler().fit(raw[is_train])(raw)
    mu, sd = log_price[is_train].mean(), log_price[is_train].std()
    y_scaled = ((log_price - mu) / sd).astype(np.float32)

    preds = {}
    preds["Comparables median"] = comps_median_baseline(log_price, g, fallback=float(np.median(log_price[is_train])))

    lgb_a = fit_lightgbm(raw[is_train], log_price[is_train])
    preds["LightGBM (features + location)"] = lgb_a.predict(raw)

    raw_c = np.column_stack([raw, comps_features(log_price, g)])
    lgb_b = fit_lightgbm(raw_c[is_train], log_price[is_train])
    preds["LightGBM + comparables features"] = lgb_b.predict(raw_c)

    model = train_gnn(x, y_scaled, g, split, epochs=args.epochs)
    gnn_scaled, weights = predict(model, x, y_scaled, g)
    preds["Comparables GNN (ours)"] = gnn_scaled * sd + mu

    actual = df["price"].to_numpy()
    results = {name: summarise(actual[is_test], np.exp(p[is_test])) for name, p in preds.items()}

    gnn = preds["Comparables GNN (ours)"]
    half = conformal_halfwidth(log_price[is_val], gnn[is_val], coverage=0.9)
    lo, hi = np.exp(gnn - half), np.exp(gnn + half)
    coverage = float(np.mean((actual[is_test] >= lo[is_test]) & (actual[is_test] <= hi[is_test])))
    width = float(np.median((hi - lo)[is_test] / np.exp(gnn[is_test])))

    summary = {
        "data": source, "sales": len(df), "test_sales": int(is_test.sum()),
        "test_period": f"{df['date'][is_test].min().date()} to {df['date'][is_test].max().date()}",
        "results": results,
        "interval": {"target_coverage_pct": 90, "test_coverage_pct": round(coverage * 100, 1),
                     "median_width_pct_of_estimate": round(width * 100, 1)},
        "gnn_mean_weight_on_own_guess": round(float(weights[is_test, 0].mean()), 3),
        "runtime_s": round(time.time() - started, 1),
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2))

    lines = [f"Data: {source}. Test set: the latest {is_test.sum()} sales ({summary['test_period']}).", "",
             "| Model | Median abs. % error | Within 10% | Within 20% |", "|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['median_abs_pct_error']}% | {r['within_10pct']}% | {r['within_20pct']}% |")
    lines += ["", f"GNN 90% interval: covered {coverage * 100:.1f}% of test sales, "
                  f"median width {width * 100:.1f}% of the estimate."]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    # Files for the demo app: each test sale, and the comparables behind its estimate.
    test_idx = np.flatnonzero(is_test)
    t = df.iloc[test_idx].copy()
    for name, p in preds.items():
        t[f"est_{name}"] = np.exp(p[test_idx]).round(-3)
    t["gnn_low"], t["gnn_high"] = lo[test_idx].round(-3), hi[test_idx].round(-3)
    t["own_guess_weight"] = weights[test_idx, 0]
    t.to_csv(out / "test_predictions.csv", index=False)

    rows = []
    for i in test_idx:
        for slot in np.flatnonzero(g.mask[i]):
            j = g.nbr[i, slot]
            rows.append({"sale_id": df["sale_id"].iat[i], "comp_sale_id": df["sale_id"].iat[j],
                         "weight": float(weights[i, slot + 1]), "dist_km": float(g.dist_km[i, slot]),
                         "days_before": int(g.days_ago[i, slot]), "comp_price": float(df["price"].iat[j]),
                         "comp_date": df["date"].iat[j].date(), "lat": df["lat"].iat[j], "lon": df["lon"].iat[j],
                         "beds": df["beds"].iat[j], "baths": df["baths"].iat[j], "type": df["type"].iat[j],
                         "land_m2": df["land_m2"].iat[j]})
    pd.DataFrame(rows).to_csv(out / "test_comparables.csv", index=False)
    torch.save(model.state_dict(), out / "gnn.pt")
    print(f"\nSaved results and demo files to {out}/ in {summary['runtime_s']} s")


if __name__ == "__main__":
    main()
