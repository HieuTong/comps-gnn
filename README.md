# Learned comparables: a GNN that values homes like a valuer

A graph neural network that estimates a home's price from nearby earlier sales, and shows
which sales it trusted. Built with PyTorch, compared honestly against strong baselines.

> **Status:** runs end to end on a **simulated** Melbourne-style market. Results below show
> the method works; they are not real-market accuracy. Real sales data is the next step.

## The idea, simply

Imagine you want to guess how much your friend's house is worth. You'd look at houses
**nearby** that **sold recently**, then adjust: "that one has an extra bedroom, so ours is
a bit cheaper." That's what property valuers call **comparables**.

This project turns that into a **graph**: every sale is a dot, and each dot is connected
by lines to nearby sales that happened **before** it. A graph neural network sends a
message along each line: "I sold for this much; adjusted for our differences, you're worth
about this much." The home also makes its own guess from its features. Then the model
decides **how much to trust each message** (attention) and takes the weighted average.

Those trust weights are the explanation: the demo shows exactly which sales drove each
estimate, like a valuer's report.

## Results (simulated data, test = latest 2,250 sales, never seen in training)

| Model | Median abs. % error | Within 10% | Within 20% |
|---|---|---|---|
| Comparables median | 24.5% | 20.8% | 42.4% |
| LightGBM (features + location) | 7.0% | 67.0% | 95.6% |
| LightGBM + comparables features | 7.01% | 66.4% | 95.2% |
| **Comparables GNN** | **6.77%** | **68.2%** | **95.9%** |

90% prediction intervals covered 87.3% of test sales, slightly under target, because the
market kept moving after the validation period that set the interval width.

**How to read this honestly:** the GNN wins, but narrowly, and on data I simulated with a
hidden street-level premium that nearby sales reveal. A small win on a simulator I wrote
is evidence the pipeline works, not that GNNs beat LightGBM on real property. Real data
decides that.

## What makes the evaluation trustworthy

- **No future leakage.** A sale only connects to sales strictly before it. Tests check
  this, and check that changing a sale's own price never changes its own estimate. I
  deliberately broke the rule once: 4 tests failed, so the tests really guard it.
- **Time-based split:** train on older sales, choose settings on the next period, test on
  the latest. Never a random split.
- **Strong baselines:** a model only counts if it beats LightGBM with location features,
  including a version given the same comparables.
- **Industry metrics:** median absolute % error and share within 10% / 20%, how automated
  valuation models are compared.
- **Uncertainty:** split conformal intervals, with coverage checked on the test period.

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q                       # 19 tests, under a minute
python scripts/run_experiment.py          # about 1-2 minutes on a laptop CPU
streamlit run app.py                      # the demo: a city map, then why one home got its price
```

### On Google Colab or Kaggle (GPU)

Training picks a GPU automatically if one exists (`pick_device` in `comps/train.py`).
In a notebook cell, from the project folder:

```bash
!pip install -q lightgbm
!python scripts/run_experiment.py --n 200000
```

At the default 15,000 sales a GPU barely helps: the model is small, and building the
graph runs on the CPU either way. It helps once you have hundreds of thousands of sales.

### Your own sales

```bash
python scripts/run_experiment.py --csv sales.csv
```

The CSV needs: `sale_id, date, lat, lon, price, beds, baths, cars, land_m2, type`
(`type` is `house`, `townhouse` or `unit`). NSW publishes property-level sales data; it
needs coordinates from an address dataset such as G-NAF. Check each dataset's licence.

## Project layout

| File | What it does |
|---|---|
| `comps/data.py` | Simulated city, real-data loader, time-based split |
| `comps/graph.py` | Builds the time-respecting comparables graph |
| `comps/model.py` | The GNN: adjusted messages plus attention |
| `comps/evaluate.py` | Baselines, metrics, conformal intervals |
| `comps/train.py` | Training with early stopping on validation |
| `comps/explain.py` | Map colours by accuracy, and lines to comparables |
| `scripts/run_experiment.py` | Runs everything, writes `outputs/` |
| `app.py` | Streamlit demo |
| `tests/test_honesty.py` | Leakage, graph, model and metric tests |
| `tests/test_explain.py` | The map's colours and lines match the data |

## Next steps

1. Real data: NSW sales with G-NAF coordinates.
2. A second message-passing layer (neighbours of neighbours) and a heterogeneous graph with
   schools and stations, each kept only if it beats the current model on validation.
3. Tune on validation with several seeds, and report the spread, not one run.
