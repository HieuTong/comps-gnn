"""Demo: every test home on a map, coloured by accuracy, and why one home got its estimate.

    streamlit run app.py      (after: python scripts/run_experiment.py)
"""
import json
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

from comps.explain import BAND_NAMES, BANDS, comparable_lines, homes_for_map

OUT = Path("outputs")
GNN = "est_Comparables GNN (ours)"
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
homes = homes_for_map(sales, GNN)

st.title("Learned comparables")
st.caption("A graph neural network that values a home from nearby earlier sales, and shows which ones it trusted.")
if "SIMULATED" in summary["data"]:
    st.warning("Simulated data: homes are placed on a real map for illustration. They are not real sales.")

with st.expander("How every model did on the test period (the latest sales, never seen in training)"):
    table = pd.DataFrame(summary["results"]).T.rename(columns={
        "median_abs_pct_error": "Median abs. % error", "within_10pct": "Within 10% (%)",
        "within_20pct": "Within 20% (%)"})
    st.dataframe(table, width="stretch")
    iv = summary["interval"]
    st.write(f"90% intervals covered **{iv['test_coverage_pct']}%** of test sales "
             f"(median width {iv['median_width_pct_of_estimate']}% of the estimate).")

# ---------- pick a home ----------
sales["label"] = (sales["beds"].astype(str) + " bed " + sales["type"] + " · sold "
                  + sales["date"].dt.strftime("%d %b %Y") + " · #" + sales["sale_id"].astype(str))
choice = st.selectbox("Value one home", sales["label"], index=0)
home = sales[sales["label"] == choice].iloc[0]
mine = comps[comps["sale_id"] == home["sale_id"]].sort_values("weight", ascending=False)

# ---------- the city map ----------
st.subheader("Every test home, coloured by how close the estimate was")
counts = homes["band"].value_counts()
cols = st.columns(len(BANDS))
for col, (_, name, _) in zip(cols, BANDS):
    col.metric(name.capitalize(), f"{counts.get(name, 0):,}",
               f"{counts.get(name, 0) / len(homes) * 100:.0f}% of homes", delta_color="off")

left, right = st.columns([3, 1])
show = left.multiselect("Show", BAND_NAMES, default=BAND_NAMES)
view = right.radio("Zoom to", ["Selected home", "Whole city"], horizontal=True)

visible = homes[homes["band"].isin(show)]
layers = [
    pdk.Layer("ScatterplotLayer", data=visible, id="homes", get_position=["lon", "lat"],
              get_fill_color="colour", get_radius=45, radius_min_pixels=2, pickable=True),
    pdk.Layer("LineLayer", data=comparable_lines(home, mine), id="comparable-lines",
              get_source_position="from", get_target_position="to",
              get_color=[30, 30, 30, 170], get_width="width", width_units="pixels"),
    # Comparables are often older training sales that aren't on the map, so mark each line's end.
    pdk.Layer("ScatterplotLayer", data=mine[["lon", "lat"]], id="comparables", get_position=["lon", "lat"],
              get_fill_color=[30, 30, 30, 220], get_radius=30, radius_min_pixels=3),
    pdk.Layer("ScatterplotLayer", data=pd.DataFrame([{"lon": home["lon"], "lat": home["lat"]}]),
              id="selected-home", get_position=["lon", "lat"], get_fill_color=[20, 20, 20, 255],
              get_line_color=[255, 255, 255, 255], stroked=True, line_width_min_pixels=2,
              get_radius=90, radius_min_pixels=7),
]
if view == "Selected home":
    centre = pdk.ViewState(latitude=float(home["lat"]), longitude=float(home["lon"]), zoom=13)
else:
    centre = pdk.ViewState(latitude=float(homes["lat"].mean()), longitude=float(homes["lon"].mean()), zoom=10)

st.pydeck_chart(pdk.Deck(
    layers=layers, initial_view_state=centre, map_provider="carto", map_style=pdk.map_styles.CARTO_LIGHT,
    tooltip={"html": "<b>{home_txt}</b><br/>Sold for {price_txt}<br/>Estimate {estimate_txt} ({error_txt})"},
))
st.caption("Big black dot: the selected home. Lines lead to the earlier sales it was valued from "
           "(small black dots); thicker means the model trusted that sale more. Hover over a coloured dot for details.")

# ---------- the selected home ----------
st.subheader("Why this home got its estimate")
a, b, c, d = st.columns(4)
a.metric("Sold for", f"${home['price']:,.0f}")
err = (home[GNN] - home["price"]) / home["price"] * 100
b.metric("GNN estimate", f"${home[GNN]:,.0f}", f"{err:+.1f}% vs sale", delta_color="off")
c.metric("90% range", f"${home['gnn_low'] / 1e6:.2f}M – ${home['gnn_high'] / 1e6:.2f}M")
d.metric("Weight on its own features", f"{home['own_guess_weight'] * 100:.0f}%",
         help="The rest of the weight went to comparable sales.")
st.write(f"{home['beds']} bed, {home['baths']} bath, {home['cars']} car {home['type']}"
         + (f" on {home['land_m2']:.0f} m²" if home["land_m2"] > 0 else ""))

if mine.empty:
    st.info("No earlier sales nearby, so this estimate comes from the home's features alone.")
else:
    shown = mine.assign(
        Trust=(mine["weight"] * 100).round(1).astype(str) + "%",
        Price=mine["comp_price"].map("${:,.0f}".format),
        Home=mine["beds"].astype(str) + " bed " + mine["type"],
        Distance=mine["dist_km"].map("{:.2f} km".format),
        Sold=mine["days_before"].astype(str) + " days earlier",
    )[["Trust", "Price", "Home", "Distance", "Sold"]]
    st.dataframe(shown, hide_index=True, width="stretch")

st.markdown("**Other models for this home**")
others = {k.replace("est_", ""): v for k, v in home.items() if k.startswith("est_")}
st.dataframe(pd.DataFrame({"Estimate": {k: f"${v:,.0f}" for k, v in others.items()}}), width="stretch")
