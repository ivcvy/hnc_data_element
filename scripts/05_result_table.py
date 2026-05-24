"""Generate tables for data elements"""

import pandas as pd
from pathlib import Path
import ast
import sys


def item_by_domain(output_dir: str):
    df = pd.read_csv(f"{output_dir}/processed_data.csv")

    item_by_entity = (
        df.groupby("entity_name")["name"]
        .apply(lambda x: sorted(x.dropna().unique().tolist()))
        .to_dict()
    )

    max_length = max(len(attrs) for attrs in item_by_entity.values())

    padded_items = {}
    for entity, attrs in item_by_entity.items():
        padded_list = attrs + [None] * (max_length - len(attrs))
        padded_items[entity] = padded_list

    items_df = pd.DataFrame(padded_items)

    items_df.to_csv(f"{output_dir}/items_by_domain.csv", index=False)


def item_by_sig(output_dir: str):
    df = pd.read_csv(f"{output_dir}/processed_data.csv")

    result_df = (
        df.loc[df["sig"].notna(), ["entity_name", "name", "sig"]]
        .copy()
        .rename(
            columns={
                "entity_name": "domain",
                "name": "data_item",
                "sig": "significant_endpoints",
            }
        )
    )

    mask = result_df["significant_endpoints"].apply(isinstance, args=(str,))
    result_df.loc[mask, "significant_endpoints"] = result_df.loc[
        mask, "significant_endpoints"
    ].apply(ast.literal_eval)

    (
        result_df.explode("significant_endpoints")
        .groupby(["significant_endpoints", "domain"])["data_item"]
        .apply(lambda x: ", ".join(sorted(set(x))))
        .to_csv(f"{output_dir}/items_by_significant_endpoints.csv")
    )


def main(output_dir: Path):
    item_by_domain(output_dir=output_dir)
    item_by_sig(output_dir=output_dir)


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
    main(output_dir=output_dir)
