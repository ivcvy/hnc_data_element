"""Load extracted data elements"""

import sys
import pandas as pd
from pathlib import Path
from mappings import (
    ANATOMICAL_MAPPING,
    HISTO_MAPPING,
    STAGE_MAPPING,
    DOI_SPECIFIC_MAPPING,
    ALL_MAPPING,
    CUSTOM_ENTITY,
    ENDPOINT_MAPPING,
)


def load_data(input_path: Path) -> pd.DataFrame:

    def _extract_item(item: dict) -> dict:
        return {
            "id": item.get("id"),
            "name": item.get("name"),
            "measurement": item.get("measurement"),
            "unit": item.get("unit"),
            "category": item.get("category"),
        }

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    all_entry = pd.read_json(input_path, lines=True)
    val_entry = all_entry[
        all_entry.data.apply(lambda d: "doi" in d and d.get("notes") == "IV")
    ]
    print(f"Total validated entries: {len(val_entry)}")

    df_data = pd.json_normalize(val_entry["data"])

    long_format_data = []

    for _, row in df_data.iterrows():
        base_data = {
            "doi": row.get("doi"),
            "documentId": row.get("documentId"),
            "publication_year": row.get("publication_year"),
            "region": row.get("region"),
            "cancer_type": row.get("cancer_type"),
            "stage": row.get("stage"),
            "inclusion_criteria": row.get("other_inclusion_criteria"),
            "exclusion_criteria": row.get("exclusion_criteria"),
            "data_source": row.get("data_source"),
            "endpoint": row.get("endpoint"),
        }

        for entity in row.get("entity") or []:
            entity_base = {
                **base_data,
                "entity_id": entity.get("id"),
                "entity_name": entity.get("name"),
            }

            for attr in entity.get("attribute") or []:
                long_format_data.append(
                    {**entity_base, **_extract_item(attr), "item_type": "attr"}
                )

                for sub in attr.get("sub") or []:
                    long_format_data.append(
                        {**entity_base, **_extract_item(sub), "item_type": "sub"}
                    )

                    for subsub in sub.get("subsub") or []:
                        long_format_data.append(
                            {
                                **entity_base,
                                **_extract_item(subsub),
                                "item_type": "subsub",
                            }
                        )

    df_long = pd.DataFrame(long_format_data)

    exclude_columns = {"documentId", "doi"}
    for col in df_long.columns:
        if col not in exclude_columns:
            df_long[col] = df_long[col].apply(
                lambda x: x.lower() if isinstance(x, str) else x
            )

    return df_long


def _iter_endpoints(endpoint) -> list:
    if endpoint is None or (isinstance(endpoint, float) and pd.isna(endpoint)):
        return []
    return endpoint


def significant_attribute_mapping(
    df: pd.DataFrame,
    mapping_dict: dict = None,
    entity_filter: str = None,
    doi: list = None,
) -> pd.DataFrame:

    def _normalize_sig_attrs(sig_attrs) -> list:
        if isinstance(sig_attrs, str):
            return [a.strip().lower() for a in sig_attrs.split(",")]
        return sig_attrs

    if entity_filter:
        affected_dois = df.loc[
            (df["entity_name"] == entity_filter) & (df["name"].isin(mapping_dict)),
            "doi",
        ].unique()
    elif doi:
        affected_dois = doi
    else:
        affected_dois = df["doi"].unique()

    affected_mask = df["doi"].isin(affected_dois)
    name_updates: dict[tuple, str] = {}

    for _, row in df.loc[affected_mask].iterrows():
        current_doi = row["doi"]

        for e in _iter_endpoints(row["endpoint"]):
            if not isinstance(e, dict) or "significant_attribute" not in e:
                continue

            attr_list = _normalize_sig_attrs(e.get("significant_attribute"))
            e["significant_attribute"] = attr_list

            if attr_list and mapping_dict and (entity_filter or doi):
                for i, sig in enumerate(attr_list):
                    if sig in mapping_dict:
                        new_sig = mapping_dict[sig]
                        attr_list[i] = new_sig
                        name_updates[(current_doi, sig)] = new_sig

    if name_updates:
        for (doi_val, old_name), new_name in name_updates.items():
            mask = (df["doi"] == doi_val) & (df["name"] == old_name)
            df.loc[mask, "name"] = new_name

    return df


def apply_mappings(
    df: pd.DataFrame,
    anatomical_mapping: dict = ANATOMICAL_MAPPING,
    histo_mapping: dict = HISTO_MAPPING,
    stage_mapping: dict = STAGE_MAPPING,
    doi_mapping: dict = DOI_SPECIFIC_MAPPING,
    entity_mapping: dict = ALL_MAPPING,
    endpoint_mapping: dict = ENDPOINT_MAPPING,
    custom_entity: dict = CUSTOM_ENTITY,
) -> pd.DataFrame:

    def _apply_substring_mapping(series: pd.Series, mapping: dict) -> pd.Series:
        result = pd.Series([None] * len(series), index=series.index)
        for substring, value in mapping.items():
            unmatched = result.isna() & series.notna()
            matched = unmatched & series.str.contains(substring, regex=False, na=False)
            result[matched] = value
        return result

    df_mapped = df.copy()

    df_mapped["anatomical_site"] = _apply_substring_mapping(
        df_mapped["cancer_type"], anatomical_mapping
    )
    df_mapped["histology"] = _apply_substring_mapping(
        df_mapped["cancer_type"], histo_mapping
    )
    df_mapped["stage"] = df_mapped["stage"].replace(stage_mapping)
    print("Completed anatomical, histology, and stage mappings.")

    for mapping_config in doi_mapping:
        doi = mapping_config["doi"]
        mapping = mapping_config["mapping"]
        df_mapped = significant_attribute_mapping(
            df=df_mapped, mapping_dict=mapping, doi=doi
        )
        mask = df_mapped["doi"].isin(doi) & df_mapped["name"].isin(mapping)
        df_mapped.loc[mask, "name"] = df_mapped.loc[mask, "name"].replace(mapping)
    print(f"Completed {len(doi_mapping)} DOI specific mappings.")

    for entity_name, mapping in entity_mapping.items():
        df_mapped = significant_attribute_mapping(
            df=df_mapped, mapping_dict=mapping, entity_filter=entity_name
        )
        mask = df_mapped["entity_name"] == entity_name
        df_mapped.loc[mask, "name"] = df_mapped.loc[mask, "name"].replace(mapping)
    df_mapped["entity_name"] = df_mapped["entity_name"].replace(
        "infrastructure", "health system"
    )
    print(f"Completed {len(entity_mapping)} entity specific mappings.")

    for _, row in df_mapped.iterrows():
        for endpoint in _iter_endpoints(row["endpoint"]):
            if isinstance(endpoint, dict):
                endpoint["name"] = endpoint_mapping.get(
                    endpoint.get("name"), endpoint.get("name")
                )
    print(f"Completed {len(endpoint_mapping)} endpoint mappings.")

    df_mapped["entity_name"] = df_mapped["entity_name"].where(
        ~df_mapped["name"].isin(custom_entity), df_mapped["name"].map(custom_entity)
    )
    print(f"Completed {len(custom_entity)} custom entity mappings.")

    return df_mapped


def add_sig(df: pd.DataFrame) -> pd.DataFrame:

    def get_sig(row):
        matches = [
            ep.get("name")
            for ep in _iter_endpoints(row["endpoint"])
            if isinstance(ep, dict)
            and row["name"] in (ep.get("significant_attribute") or [])
        ]
        return matches if matches else None

    df["sig"] = df.apply(get_sig, axis=1)
    df["sig"] = df["sig"].replace({None: "nan"})
    return df


def main(input_path: Path, output_dir: Path):
    df = load_data(input_path=input_path)
    df_mapped = apply_mappings(df=df)
    df_mapped = add_sig(df=df_mapped)
    df_mapped.to_csv(f"{output_dir}/processed_data.csv", index=False)


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/entities_00001.jsonl"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    main(input_path=input_path, output_dir=output_dir)
