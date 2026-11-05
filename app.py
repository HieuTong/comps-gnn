"""Demo: pick a test sale and see how the GNN valued it, and which sales it trusted.

    streamlit run app.py      (after: python scripts/run_experiment.py)
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

OUT = Path("outputs")
st.set_page_config(page_title="Learned comparables", layout="wide")

if not (OUT / "results.json").exists():
    st.error("No results yet. Run: python scripts/run_experiment.py")
    st.stop()


@st.cache_data
def load():
    summary = json.loads((OUT / "results.json").read_text())
    sales = pd.read_csv(OUT / "test_predictions.csv", parse_dates=["date"])
    comps = pd.read_csv(OUT / "test_comparables.csv")
    return summary, sales, comps


summary, sales, comps = load()
GNN = "est_Comparables GNN (ours)"

st.title("Learned comparables")
st.caption("A graph neural network that values a home from nearby earlier sales, and shows which ones it trusted.")
if "SIMULATED" in summary["data"]:
    st.warning("Simulated data: these results show the method works end to end, not real-market accuracy.")

with st.expander("How every model did on the test period (the latest sales, never seen in training)", expanded=False):
    table = pd.DataFrame(summary["results"]).T.rename(columns={
        "median_abs_pct_error": "Median abs. % error", "within_10pct": "Within 10% (%)",
        "within_20pct": "Within 20% (%)"})
    st.dataframe(table, width="stretch")
    iv = summary["interval"]
    st.write(f"90% intervals covered **{iv['test_coverage_pct']}%** of test sales "
             f"(median width {iv['median_width_pct_of_estimate']}% of the estimate).")

st.subheader("Value one home")
sales["label"] = (sales["beds"].astype(str) + " bed " + sales["type"] + " · sold "
                  + sales["date"].dt.strftime("%d %b %Y") + " · #" + sales["sale_id"].astype(str))
choice = st.selectbox("Test sale", sales["label"], index=0)
home = sales[sales["label"] == choice].iloc[0]

a, b, c, d = st.columns(4)
a.metric("Sold for", f"${home['price']:,.0f}")
err = (home[GNN] - home["price"]) / home["price"] * 100
b.metric("GNN estimate", f"${home[GNN]:,.0f}", f"{err:+.1f}% vs sale", delta_color="off")
c.metric("90% range", f"${home['gnn_low'] / 1e6:.2f}M – ${home['gnn_high'] / 1e6:.2f}M")
d.metric("Weight on its own features", f"{home['own_guess_weight'] * 100:.0f}%",
         help="The rest of the weight went to comparable sales.")

st.write(f"{home['beds']} bed, {home['baths']} bath, {home['cars']} car {home['type']}"
         + (f" on {home['land_m2']:.0f} m²" if home["land_m2"] > 0 else ""))

mine = comps[comps["sale_id"] == home["sale_id"]].sort_values("weight", ascending=False)
if mine.empty:
    st.info("No earlier sales nearby, so this estimate comes from the home's features alone.")
else:
    st.markdown("**The sales the model trusted most**")
    shown = mine.assign(
        Trust=(mine["weight"] * 100).round(1).astype(str) + "%",
        Price=mine["comp_price"].map("${:,.0f}".format),
        Distance=mine["dist_km"].map("{:.2f} km".format),
        **{"Sold": mine["days_before"].astype(str) + " days earlier"},
        Home=mine["beds"].astype(str) + " bed " + mine["type"],
    )[["Trust", "Price", "Home", "Distance", "Sold"]]
    st.dataframe(shown, hide_index=True, width="stretch")

    pts = pd.concat([
        pd.DataFrame({"lat": [home["lat"]], "lon": [home["lon"]], "size": [60.0], "color": ["#d1495b"]}),
        pd.DataFrame({"lat": mine["lat"], "lon": mine["lon"], "size": 15 + 200 * mine["weight"],
                      "color": "#2e7d32"}),
    ])
    st.map(pts, latitude="lat", longitude="lon", size="size", color="color")
    st.caption("Red: the home being valued. Green: its comparables, bigger = trusted more.")

st.markdown("**Other models for this home**")
others = {k.replace("est_", ""): v for k, v in home.items() if k.startswith("est_")}
st.dataframe(pd.DataFrame({"Estimate": {k: f"${v:,.0f}" for k, v in others.items()}}), width="stretch")
