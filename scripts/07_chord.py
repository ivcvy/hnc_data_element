"""Generate chord diagram"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict, Counter
from ast import literal_eval
from pathlib import Path
import sys


OUTER_COLORS = ["#D73027", "#377EB8", "#09A129", "#b30089", "#FF7F00", "#838383"]
INNER_BASE, INNER_OUTER = 1.0, 1.3
OUTER_BASE, OUTER_OUTER = 1.6, 1.9
DOMAIN_GAP = 0.02
INNER_GAP = 0.01
ENDPOINT_GAP = 0.02
ENDPOINT_WIDTH = 0.272
LABEL_RADIUS = OUTER_OUTER + 0.25


def load_data(output_dir: str) -> pd.DataFrame:
    df = pd.read_csv(f"{output_dir}/processed_data.csv")[
        ["doi", "entity_name", "name", "sig"]
    ].rename(
        columns={
            "entity_name": "domain",
            "name": "data_item",
            "sig": "significant_endpoint",
        }
    )
    df["significant_endpoint"] = df["significant_endpoint"].apply(
        lambda x: (
            literal_eval(x)
            if isinstance(x, str)
            else (x if isinstance(x, list) else [])
        )
    )
    return df


def sort_and_explode(df: pd.DataFrame) -> pd.DataFrame:
    all_endpoint_counts = Counter(
        ep for lst in df["significant_endpoint"] for ep in lst
    )

    def sort_key(row):
        eps = row["significant_endpoint"]
        if not eps:
            return (0, "")
        return (max(all_endpoint_counts[ep] for ep in eps), min(str(ep) for ep in eps))

    return (
        df.assign(sort_key=df.apply(sort_key, axis=1))
        .sort_values(["domain", "sort_key", "data_item"], ascending=[True, False, True])
        .drop(columns="sort_key")
        .reset_index(drop=True)
        .reset_index()
        .explode("significant_endpoint")
    )


def build_pairs(sorted_df: pd.DataFrame) -> tuple[dict, dict]:
    all_pairs: dict = defaultdict(lambda: defaultdict(list))
    for _, row in sorted_df.iterrows():
        ep = row["significant_endpoint"]
        ep_clean = ep if (pd.notna(ep) and ep != "") else None
        all_pairs[row["domain"]][row["index"]].append(
            (row["data_item"], ep_clean, row["domain"])
        )

    max_id = sorted_df["index"].max() if not sorted_df.empty else 0
    unique_endpoints = sorted_df["significant_endpoint"].dropna().unique().tolist()
    endpoint_lookup = {ep: max_id + i + 1 for i, ep in enumerate(unique_endpoints)}

    all_pairs["endpoint"] = {
        ep_id: [(ep, None, "endpoint")] for ep, ep_id in endpoint_lookup.items()
    }

    return {k: dict(v) for k, v in all_pairs.items()}, endpoint_lookup


def compute_domain_bounds(domains: list, all_pairs: dict) -> dict:
    normal_domains = [d for d in domains if d != "endpoint"]
    total_normal = sum(len(all_pairs[d]) for d in normal_domains)
    available = 2 * np.pi - ENDPOINT_WIDTH

    portions = {d: len(all_pairs[d]) / total_normal * available for d in normal_domains}
    portions["endpoint"] = ENDPOINT_WIDTH

    bounds, current = {}, 0.0
    for d in domains:
        bounds[d] = (current + DOMAIN_GAP / 2, current + portions[d] - DOMAIN_GAP / 2)
        current += portions[d]
    return bounds


def assign_equal_slots(ids, a1, a2, gap, item_start_angle, item_end_angle, item_angles):
    n = len(ids)
    available = a2 - a1 - n * gap
    boundaries = np.linspace(a1, a1 + available, n + 1)
    for j, id_ in enumerate(ids):
        start = boundaries[j] + j * gap
        end = boundaries[j + 1] + j * gap
        item_start_angle[id_] = start
        item_end_angle[id_] = end
        item_angles[id_] = (start + end) / 2


def compute_angles(domains, all_pairs, domain_bounds, endpoint_lookup):
    item_angles, item_start_angle, item_end_angle = {}, {}, {}
    normal_domains = [d for d in domains if d != "endpoint"]

    endpoint_ribbon_counts: dict[int, int] = defaultdict(int)
    for d in normal_domains:
        for tuples in all_pairs[d].values():
            for _, ep, _ in tuples:
                if ep and ep in endpoint_lookup:
                    endpoint_ribbon_counts[endpoint_lookup[ep]] += 1

    for domain in domains:
        ids = list(all_pairs[domain])
        a1, a2 = domain_bounds[domain]

        if domain == "endpoint":
            ids_sorted = sorted(
                ids, key=lambda i: endpoint_ribbon_counts[i], reverse=True
            )
            total_ribbons = sum(endpoint_ribbon_counts[i] for i in ids_sorted)
            remaining = a2 - a1 - (len(ids_sorted) + 1) * ENDPOINT_GAP
            current = a1 + ENDPOINT_GAP
            for id_ in ids_sorted:
                width = (
                    remaining * endpoint_ribbon_counts[id_] / total_ribbons
                    if total_ribbons
                    else 0
                )
                item_start_angle[id_] = current
                item_end_angle[id_] = current + width
                item_angles[id_] = current + width / 2
                current += width + ENDPOINT_GAP
        else:
            assign_equal_slots(
                ids, a1, a2, INNER_GAP, item_start_angle, item_end_angle, item_angles
            )

    endpoint_sources: dict[int, list] = defaultdict(list)
    for d in normal_domains:
        for src_id, tuples in all_pairs[d].items():
            for _, ep, _ in tuples:
                if ep and ep in endpoint_lookup:
                    endpoint_sources[endpoint_lookup[ep]].append(src_id)

    ribbon_angles = {}
    for ep_id, src_ids in endpoint_sources.items():
        angles = np.linspace(
            item_start_angle[ep_id], item_end_angle[ep_id], len(src_ids) + 2
        )[1:-1]
        for src_id, ang in zip(src_ids, angles):
            ribbon_angles[(src_id, ep_id)] = ang

    return item_angles, item_start_angle, item_end_angle, ribbon_angles


def polar_to_cartesian(angle, radius):
    return radius * np.cos(angle), radius * np.sin(angle)


def ring_segment(a1, a2, r_inner, r_outer, n=30):
    theta_out = np.linspace(a1, a2, n)
    theta_in = np.linspace(a2, a1, n)
    xo, yo = polar_to_cartesian(theta_out, r_outer)
    xi, yi = polar_to_cartesian(theta_in, r_inner)
    return np.concatenate([xo, xi]), np.concatenate([yo, yi])


def bezier_ribbon(a1, a2, r_top, curvature=0.45, n=100):
    x0, y0 = polar_to_cartesian(a1, r_top)
    x1, y1 = polar_to_cartesian(a2, r_top)
    xm = (x0 + x1) / 2 * (1 - curvature)
    ym = (y0 + y1) / 2 * (1 - curvature)
    t = np.linspace(0, 1, n)
    x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * xm + t**2 * x1
    y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * ym + t**2 * y1
    return x, y


def build_figure(
    domains,
    all_pairs,
    domain_bounds,
    domain_to_color,
    item_angles,
    item_start_angle,
    item_end_angle,
    ribbon_angles,
    endpoint_lookup,
):
    fig = go.Figure()
    normal_domains = [d for d in domains if d != "endpoint"]

    # Inner ring
    for domain in domains:
        color = domain_to_color[domain]
        for id_, tuples in all_pairs[domain].items():
            has_endpoint = any(t[1] is not None for t in tuples)
            seg_color = color if (domain == "endpoint" or has_endpoint) else "lightgray"
            x, y = ring_segment(
                item_start_angle[id_], item_end_angle[id_], INNER_BASE, INNER_OUTER
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    fill="toself",
                    fillcolor=seg_color,
                    line=dict(color="white", width=0.4),
                    hovertemplate=f"<b>{domain}</b><extra></extra>",
                    showlegend=False,
                )
            )

    # Outer ring
    for i, domain in enumerate(domains):
        a1, a2 = domain_bounds[domain]
        x, y = ring_segment(
            a1 + DOMAIN_GAP / 2, a2 - DOMAIN_GAP / 2, OUTER_BASE, OUTER_OUTER
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                fill="toself",
                fillcolor=OUTER_COLORS[i % len(OUTER_COLORS)],
                line=dict(color="white", width=1),
                opacity=0.4,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Ribbons
    for domain in normal_domains:
        color = domain_to_color[domain]
        for src_id, tuples in all_pairs[domain].items():
            for item, endpoint, _ in tuples:
                if not endpoint or endpoint not in endpoint_lookup:
                    continue
                ep_id = endpoint_lookup[endpoint]
                tgt_angle = ribbon_angles.get((src_id, ep_id), item_angles[ep_id])
                x, y = bezier_ribbon(item_angles[src_id], tgt_angle, r_top=INNER_BASE)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        line=dict(color=color, width=2),
                        opacity=0.1,
                        hoverinfo="text",
                        hovertext=f"Item: {item}, Endpoint: {endpoint}",
                        showlegend=False,
                    )
                )

    # Domain labels
    for domain in domains:
        mid = sum(domain_bounds[domain]) / 2
        lx, ly = polar_to_cartesian(mid, LABEL_RADIUS)
        fig.add_annotation(
            x=lx,
            y=ly,
            text=domain.capitalize(),
            showarrow=False,
            font=dict(size=11, color="black"),
            xanchor="center",
            yanchor="middle",
        )

    fig.update_layout(
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(
            visible=False, showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        width=950,
        height=950,
        margin=dict(t=60, b=60, l=60, r=60),
    )
    return fig


def main(output_dir: Path):
    df = load_data(output_dir=output_dir)
    sorted_df = sort_and_explode(df)

    domains = sorted_df["domain"].unique().tolist() + ["endpoint"]
    domain_to_color = {
        d: OUTER_COLORS[i % len(OUTER_COLORS)] for i, d in enumerate(domains)
    }

    all_pairs, endpoint_lookup = build_pairs(sorted_df)
    domain_bounds = compute_domain_bounds(domains, all_pairs)
    item_angles, item_start_angle, item_end_angle, ribbon_angles = compute_angles(
        domains, all_pairs, domain_bounds, endpoint_lookup
    )

    fig = build_figure(
        domains,
        all_pairs,
        domain_bounds,
        domain_to_color,
        item_angles,
        item_start_angle,
        item_end_angle,
        ribbon_angles,
        endpoint_lookup,
    )

    fig.show()
    fig.write_image(f"{output_dir}/chord_diagram_raw.svg")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
    main(output_dir=output_dir)
