"""Generate bubble plot"""

import textwrap
from ast import literal_eval
import numpy as np
import pandas as pd
from bokeh.io import output_file, save
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from mappings import SOURCE_MAPPING
from pathlib import Path
import sys


COLOR_MAP = {
    "national database": "#09A129",
    "cancer registry, national database, administrative and claims database": "#636B2F",
    "cancer registry": "#09A129",
    "healthcare systems or platforms": "#0014A8",
    "medical record, cancer registry": "#069494",
    "collaborative chart review, patient reported outcomes": "#0014A8",
    "single center database": "#0014A8",
    "administrative and claims database": "#D20A2E",
    "multiple centers database": "#0014A8",
    "collaborative chart review": "#0014A8",
    "cancer registry, administrative and claims database, death registry, census, "
    "healthcare systems or platforms, multiple centers database": "#000000",
    "medical record, patient reported outcomes": "#0014A8",
    "medical record": "#0014A8",
    "healthcare systems or platforms, medical record, death registry": "#b30089",
    "cancer registry, medical record": "#069494",
    "case report form": "#0014A8",
}
FALLBACK_COLOR = "#7f7f7f"
JITTER_SEED = 42
JITTER_STD = 0.15
BUBBLE_MIN_SIZE, BUBBLE_RANGE = 8, 25
LABEL_WRAP_WIDTH = 45


def parse_num_years(period) -> int:
    if not period or not isinstance(period, str):
        return 1
    total = 0
    for token in period.split(","):
        token = token.strip()
        if "-" in token:
            try:
                start, end = map(int, token.split("-"))
                total += end - start + 1
            except ValueError:
                print(f"Invalid period range: {token}")
        else:
            try:
                int(token)
                total += 1
            except ValueError:
                print(f"Invalid period token: {token}")
    return total


def expand_row(row: dict) -> list[dict]:
    raw = row.get("data_source")
    if pd.isna(raw):
        return []
    entries = literal_eval(raw)
    if isinstance(entries, dict):
        entries = [entries]
    records = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        source_type = entry.get("source_type")
        if not source_type:
            continue
        if source_type not in SOURCE_MAPPING:
            raise ValueError(f"Unknown source type: {source_type}")
        mapped = SOURCE_MAPPING[source_type]
        mapped_str = ", ".join(mapped) if isinstance(mapped, list) else mapped
        period = entry.get("period")
        records.append(
            {
                "original_source_type": source_type,
                "source_type": mapped_str,
                "sample_size": entry.get("sample_size"),
                "period": period,
                "num_years": parse_num_years(period),
            }
        )
    return records


def load_and_expand(output_dir: str) -> pd.DataFrame:
    df = pd.read_csv(f"{output_dir}/processed_data.csv").drop_duplicates("doi")
    records = [rec for row in df.to_dict("records") for rec in expand_row(row)]
    source_df = pd.DataFrame(records)
    source_df["sample_size"] = pd.to_numeric(source_df["sample_size"], errors="coerce")
    return (
        source_df.dropna(subset=["sample_size"])
        .query("num_years > 0")
        .reset_index(drop=True)
    )


def compute_y_positions(plot_data: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    source_types = (
        plot_data.groupby("source_type")["sample_size"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    n = len(source_types)
    return source_types, {src: n - 1 - i for i, src in enumerate(source_types)}


def compute_bubble_sizes(num_years: pd.Series) -> list[float]:
    lo, hi = num_years.min(), num_years.max()
    year_range = hi - lo
    scale = (
        lambda y: (
            BUBBLE_MIN_SIZE
            + (0 if year_range == 0 else (y - lo) / year_range * BUBBLE_RANGE)
        )
        * 2
    )
    return num_years.map(scale).tolist()


def compute_y_vals(plot_data: pd.DataFrame, y_pos_map: dict[str, int]) -> list[float]:
    np.random.seed(JITTER_SEED)
    jitter = np.random.normal(0, JITTER_STD, len(plot_data))
    return [
        y_pos_map[src] + jitter[i] for i, src in enumerate(plot_data["source_type"])
    ]


def build_figure(plot_data: pd.DataFrame) -> figure:
    source_types, y_pos_map = compute_y_positions(plot_data)
    n = len(source_types)

    cds = ColumnDataSource(
        data=dict(
            x=plot_data["sample_size"],
            y=compute_y_vals(plot_data, y_pos_map),
            size=compute_bubble_sizes(plot_data["num_years"]),
            source_type=plot_data["source_type"],
            original_source_type=plot_data["original_source_type"],
            num_years=plot_data["num_years"],
            period=plot_data["period"].fillna("N/A"),
        )
    )

    palette = [COLOR_MAP.get(st.lower(), FALLBACK_COLOR) for st in source_types]
    color_map = factor_cmap("source_type", palette=palette, factors=source_types)

    p = figure(
        width=1200,
        height=max(400, n * 80),
        title="Distribution of sample size and number of years covered by data source type",
        x_axis_label="Sample Size",
        y_axis_label="Data Source",
        x_axis_type="log",
        sizing_mode="stretch_width",
        toolbar_location="above",
    )
    p.min_border_left = 350
    p.min_border_right = 100
    p.min_border_top = 60
    p.min_border_bottom = 80

    scatter = p.circle(
        "x",
        "y",
        size="size",
        alpha=0.2,
        color=color_map,
        source=cds,
        line_color="white",
        line_width=1,
    )

    p.add_tools(
        HoverTool(
            tooltips=[
                ("Source Type", "@source_type"),
                ("Original Source Type", "@original_source_type"),
                ("Sample Size", "@x{0,0}"),
                ("Coverage Duration", "@num_years years"),
                ("Period", "@period"),
            ],
            renderers=[scatter],
        )
    )

    p.xaxis.formatter = NumeralTickFormatter(format="0,0")
    p.xaxis.major_label_standoff = 15
    p.xaxis.major_tick_line_width = 2
    p.xaxis.minor_tick_line_width = 1

    p.yaxis.ticker = list(range(n))
    p.yaxis.major_label_overrides = {
        n - 1 - i: "\n".join(textwrap.wrap(st.capitalize(), width=LABEL_WRAP_WIDTH))
        for i, st in enumerate(source_types)
    }
    p.yaxis.major_label_text_font_size = "9pt"
    p.yaxis.major_label_standoff = 15
    p.grid.grid_line_alpha = 0.3

    return p


def main(output_dir: Path):
    plot_data = load_and_expand(output_dir=output_dir)

    if plot_data.empty:
        print("No plottable data found.")
    else:
        output_file(f"{output_dir}/sample_size_vs_data_source_bubble.html")
        save(build_figure(plot_data=plot_data))


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
    main(output_dir=output_dir)
