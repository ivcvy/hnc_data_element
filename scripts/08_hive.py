"""Generate hive plots"""

import networkx as nx
from networkx.algorithms import bipartite
import community as community_louvain
from pathlib import Path
import pandas as pd
from hiveplotlib import NodeCollection
from hiveplotlib.edges import Edges
from hiveplotlib.hiveplot import HivePlot
import sys


STANDARDS = ["HeNeCOn", "OMOP", "O3", "mCODE"]
DOMAINS = ["diagnosis", "health system", "person", "treatment", "tumor"]

DOMAIN_COLORS = {
    "diagnosis": "#D20A2E",
    "health system": "#0014A8",
    "person": "#09A129",
    "treatment": "#b30089",
    "tumor": "#FF7518",
}
NODE_PARTITION_COLORS = {
    f"{domain}_{suffix}": color
    for domain, color in DOMAIN_COLORS.items()
    for suffix in ("obs", "mapped")
}
INTER_EDGE_COLORS = {
    f"{domain}_obs": {f"{domain}_mapped": color}
    for domain, color in DOMAIN_COLORS.items()
}


def get_data(output_dir: Path, std: str, domain: str) -> pd.DataFrame:
    """Filter data and cluster observed nodes by shared community of mapped targets."""
    df = pd.read_csv(f"{output_dir}/appendix_valid.csv")
    df = df[(df["type"] == std) & (df["domain"] == domain)]

    G = nx.Graph()
    G.add_nodes_from(df["name"], bipartite="observed")
    G.add_nodes_from(df["target_id"], bipartite="target")
    G.add_edges_from(zip(df["name"], df["target_id"]))

    observed_nodes = [n for n, d in G.nodes(data=True) if d["bipartite"] == "observed"]
    obs_graph = bipartite.weighted_projected_graph(G, observed_nodes)
    partition = community_louvain.best_partition(obs_graph, weight="weight")

    df["cluster"] = df["name"].map(partition)
    return df.sort_values(by=["cluster", "name", "target_id"])


def node_data(input_data: pd.DataFrame) -> pd.DataFrame:
    df_obs = input_data.drop_duplicates(subset=["name"]).copy()
    df_obs[["target_id", "target_description"]] = ""
    df_obs["axis"] = df_obs["domain"].astype(str) + "_obs"

    df_target = (
        input_data.drop_duplicates(subset=["target_id"])
        .dropna(subset=["target_id", "target_description"], how="all")
        .copy()
    )
    df_target[["name", "category", "measurement", "unit"]] = ""
    df_target["axis"] = df_target["domain"].astype(str) + "_mapped"

    return (
        pd.concat([df_obs, df_target], ignore_index=True)
        .reset_index()
        .rename(columns={"index": "unique_id"})
    )


def build_node_collection(output_dir: Path, std: str, domain: str) -> NodeCollection:
    original_df = get_data(output_dir=output_dir, std=std, domain=domain)
    nc = NodeCollection(data=node_data(original_df), unique_id_column="unique_id")
    nc.original_df = original_df
    return nc


def build_edges(nodes: NodeCollection) -> Edges:
    uid_col = nodes.unique_id_column
    name_to_id = nodes.data.set_index("name")[uid_col].to_dict()
    target_to_id = (
        nodes.data[nodes.data["axis"].str.endswith("_mapped")]
        .set_index("target_id")[uid_col]
        .to_dict()
    )
    edge_df = pd.DataFrame(
        {
            "from": nodes.original_df["name"].map(name_to_id),
            "to": nodes.original_df["target_id"].map(target_to_id),
        }
    )
    return Edges(data=edge_df, from_column_name="from", to_column_name="to")


def build_hive_plot(nodes: NodeCollection, edges: Edges) -> HivePlot:
    hp = HivePlot(
        nodes=nodes,
        edges=edges,
        partition_variable="axis",
        sorting_variables="unique_id",
        backend="plotly",
    )

    hp.nodes.data["node_color"] = hp.nodes.data[hp.partition_variable].map(
        NODE_PARTITION_COLORS
    )

    unmapped_ids = hp.edges.data[hp.edges.data["to"].isna()]["from"].unique()
    hp.nodes.data.loc[
        hp.nodes.data[hp.nodes.unique_id_column].isin(unmapped_ids), "node_color"
    ] = "#838383"

    hp.update_partition_data()
    hp.update_node_viz_kwargs(color="node_color")

    for p1, targets in INTER_EDGE_COLORS.items():
        for p2, color in targets.items():
            if p1 in hp.axes and p2 in hp.axes:
                hp.update_edges(partition_id_1=p1, partition_id_2=p2, color=color)

    return hp


def main(output_dir: Path):
    for std in STANDARDS:
        for domain in DOMAINS:
            nodes = build_node_collection(output_dir=output_dir, std=std, domain=domain)
            edges = build_edges(nodes)
            hp = build_hive_plot(nodes, edges)

            fig = hp.plot(node_kwargs={"size": 5}, line_width=1, opacity=0.1)
            fig.update_layout(width=4000, height=2000)

            base = f"{output_dir}/hive_plot_{std}_{domain}"
            fig.write_image(f"{base}.svg")
            fig.write_html(f"{base}.html", include_plotlyjs="cdn", full_html=True)


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
    main(output_dir=output_dir)
