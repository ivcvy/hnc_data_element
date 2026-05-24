"""Generate summary characteristics of included papers"""

from ast import literal_eval
from pathlib import Path
from mappings import (
    REGION_MAPPING,
    SOURCE_MAPPING,
    INCLUSION_MAPPING,
    EXCLUSION_MAPPING,
)
import pandas as pd
from collections import Counter
import sys


def get_decade(year: int) -> str:
    return f"{(year // 10) * 10}s"


def create_summary_row(
    df: pd.DataFrame,
    column: str,
    label: str,
    n_doi: int,
    rounding: int,
    multiple: bool = False,
) -> pd.DataFrame:
    count = df[column].notna().sum()
    if multiple:
        count = df[column].apply(lambda x: len(x) > 0).sum()
    return pd.DataFrame(
        {"count": [count], "%": [round(count / n_doi * 100, rounding)]}, index=[label]
    )


def count_values(df: pd.DataFrame, column: str) -> Counter:
    return Counter(
        item
        for item_list in df[column]
        if isinstance(item_list, list)
        for item in item_list
    )


def create_percentage_row(
    df: pd.DataFrame, count_col: str, n_doi: int, rounding: int, new_col="%"
):
    df[new_col] = round(df[count_col] / n_doi * 100, rounding)
    return df[new_col]


def parse_comma_separated(value, transform_func=None):
    if pd.isna(value):
        return []
    items = [item.strip() for item in str(value).split(",")]
    return [transform_func(item) for item in items] if transform_func else items


def parse_period(period):
    years = []
    if isinstance(period, str):
        for p in period.split(","):
            p = p.strip()
            if "-" in p:
                start, end = map(int, p.split("-"))
                years.extend(range(start, end + 1))
            else:
                years.append(int(p))
    else:
        years.append(int(period))
    return years


def preprocess_dataframe(
    df: pd.DataFrame,
    summary: bool = True,
):
    df = df.copy()
    df = df.drop_duplicates("doi").copy()
    df["data_source"] = df["data_source"].apply(literal_eval)
    df["endpoint"] = df["endpoint"].apply(literal_eval)

    df["region_parsed"] = df["region"].apply(
        lambda x: parse_comma_separated(x, str.title) if pd.notna(x) else []
    )

    if summary:
        df["region_mapped"] = df["region_parsed"].apply(
            lambda regions: [REGION_MAPPING.get(r, "Other") for r in regions]
        )

    df["anatomical_site_parsed"] = df["anatomical_site"].apply(
        lambda x: parse_comma_separated(x, str.capitalize) if pd.notna(x) else []
    )

    def extract_source_types(sources):
        if not isinstance(sources, list) or len(sources) == 0:
            return []
        source_types = []
        for source in sources:
            if isinstance(source, dict):
                source_type = source.get("source_type")
                if source_type and source_type.strip():
                    source_types.append(source_type.strip())
        return source_types

    df["data_source_types"] = df["data_source"].apply(extract_source_types)

    if summary:

        def map_source_types(types):
            if not types:
                return []
            mapped = []
            for t in types:
                mapped_value = SOURCE_MAPPING.get(t, "Other")
                if isinstance(mapped_value, list):
                    mapped.extend(mapped_value)
                else:
                    mapped.append(mapped_value)
            return mapped

        df["data_source_mapped"] = df["data_source_types"].apply(map_source_types)

    def extract_years(sources):
        if not isinstance(sources, list) or len(sources) == 0:
            return []
        years = []
        for source in sources:
            if isinstance(source, dict) and source.get("period"):
                years.extend(parse_period(source.get("period")))
        return sorted(set(years))

    df["data_period_years"] = df["data_source"].apply(extract_years)

    if summary:
        df["data_period_decades"] = df["data_period_years"].apply(
            lambda years: sorted(set(get_decade(y) for y in years)) if years else []
        )
        df["publication_decade"] = df["publication_year"].apply(
            lambda y: get_decade(int(y)) if pd.notna(y) else None
        )

    def extract_endpoint_names(endpoints):
        if not isinstance(endpoints, list) or len(endpoints) == 0:
            return []
        return [
            e.get("name") for e in endpoints if isinstance(e, dict) and e.get("name")
        ]

    df["endpoint_names"] = df["endpoint"].apply(extract_endpoint_names)

    return df


def process_criteria(
    csv_path: str,
    label_column: str,
    criteria_name: str,
    label_mapping: dict,
    rounding: int,
):
    criteria_labels = pd.read_csv(
        csv_path, sep=",", quotechar='"', skipinitialspace=True, engine="python"
    )
    criteria_labels[label_column] = criteria_labels[label_column].str.split(",")

    row_criteria = create_summary_row(
        criteria_labels,
        label_column,
        criteria_name,
        len(criteria_labels),
        rounding,
    )

    criteria_labels[label_column] = criteria_labels[label_column].apply(
        lambda x: (
            [label_mapping.get(item.strip(), item.strip()) for item in x]
            if isinstance(x, list)
            else x
        )
    )
    criteria_count = count_values(criteria_labels, label_column)
    df_criteria = (
        pd.Series(criteria_count).sort_values(ascending=False).to_frame("count")
    )
    df_criteria.index = df_criteria.index.str.capitalize()
    df_criteria["%"] = create_percentage_row(
        df_criteria, "count", len(criteria_labels), rounding
    )

    return row_criteria, df_criteria


def study_chars(
    df: pd.DataFrame,
    rounding: int,
    output_dir: Path,
    summary: bool = True,
    save_preprocessed: bool = True,
    inclusion_dir="data/inclusion_criteria_label.csv",
    exclusion_dir="data/exclusion_criteria_label.csv",
):
    df = preprocess_dataframe(df, summary=summary)
    n_doi = len(df)

    if save_preprocessed:
        preprocessed_path = f"{output_dir}/doi_study_chars.csv"
        pre_df = df[
            [
                "doi",
                "publication_year",
                "region_parsed",
                "anatomical_site",
                "histology",
                "stage",
            ]
        ].copy()
        pre_df["region_parsed"] = pre_df["region_parsed"].str.join(", ")
        pre_df.to_csv(preprocessed_path, index=False)
        print(f"Preprocessed data saved to: {preprocessed_path}")

    result = pd.DataFrame()

    # Publication year
    row_year = create_summary_row(
        df, "publication_year", "Publication year", n_doi, rounding
    )
    year_col = "publication_decade" if summary else "publication_year"

    df_year = df[year_col].value_counts().sort_index().to_frame("count")
    df_year["%"] = create_percentage_row(df_year, "count", n_doi, rounding)
    result = pd.concat([result, row_year, df_year])

    # Region
    row_region = create_summary_row(df, "region", "Region", n_doi, rounding)
    region_col = "region_mapped" if summary else "region_parsed"

    region_counts = count_values(df, region_col)
    df_region = pd.Series(region_counts).sort_values(ascending=False).to_frame("count")
    df_region["%"] = create_percentage_row(df_region, "count", n_doi, rounding)
    result = pd.concat([result, row_region, df_region])

    # Anatomical site
    row_anat = create_summary_row(
        df, "anatomical_site", "Anatomical site", n_doi, rounding
    )

    anat_count = count_values(df, "anatomical_site_parsed")
    df_anat = pd.Series(anat_count).sort_values(ascending=False).to_frame("count")
    df_anat["%"] = create_percentage_row(df_anat, "count", n_doi, rounding)
    result = pd.concat([result, row_anat, df_anat])

    # Histology
    row_hist = create_summary_row(df, "histology", "Histology", n_doi, rounding)
    df_hist = (
        df["histology"]
        .str.capitalize()
        .value_counts()
        .sort_values(ascending=False)
        .to_frame("count")
    )
    df_hist["%"] = create_percentage_row(df_hist, "count", n_doi, rounding)
    result = pd.concat([result, row_hist, df_hist])

    # Stage
    row_stage = create_summary_row(df, "stage", "Stage", n_doi, rounding)
    stage_order = {
        "Early": 1,
        "Locally advanced": 2,
        "Advanced": 3,
        "Non metastatic": 4,
        "Metastatic": 5,
    }
    df_stage = df["stage"].str.capitalize().value_counts().to_frame("count")
    df_stage["sort_key"] = df_stage.index.map(stage_order)
    df_stage = df_stage.sort_values(by="sort_key").drop(columns="sort_key")
    df_stage["%"] = create_percentage_row(df_stage, "count", n_doi, rounding)
    result = pd.concat([result, row_stage, df_stage])

    # Data source
    row_source = create_summary_row(
        df, "data_source_types", "Data source", n_doi, rounding, multiple=True
    )
    source_col = "data_source_mapped" if summary else "data_source_types"

    source_count = count_values(df, source_col)
    df_data_source = (
        pd.Series(source_count).sort_values(ascending=False).to_frame("count")
    )
    df_data_source.index = df_data_source.index.str.capitalize()
    df_data_source["%"] = create_percentage_row(
        df_data_source, "count", n_doi, rounding
    )
    result = pd.concat([result, row_source, df_data_source])

    # Data period
    row_period = create_summary_row(
        df, "data_period_years", "Data period", n_doi, rounding, multiple=True
    )
    period_col = "data_period_decades" if summary else "data_period_years"

    period_count = count_values(df, period_col)
    df_data_period = pd.Series(period_count).sort_index().to_frame("count")
    df_data_period["%"] = create_percentage_row(
        df_data_period, "count", n_doi, rounding
    )
    result = pd.concat([result, row_period, df_data_period])

    # Endpoint
    row_endpoint = create_summary_row(
        df, "endpoint_names", "Endpoint", n_doi, rounding, multiple=True
    )

    endpoint_count = count_values(df, "endpoint_names")
    df_endpoint = (
        pd.Series(endpoint_count).sort_values(ascending=False).to_frame("count")
    )
    df_endpoint.index = df_endpoint.index.str.capitalize()
    df_endpoint["%"] = create_percentage_row(df_endpoint, "count", n_doi, rounding)
    result = pd.concat([result, row_endpoint, df_endpoint])

    # Inclusion criteria
    row_inclusion, df_inclusion_labels = process_criteria(
        inclusion_dir, "label", "Inclusion criteria", INCLUSION_MAPPING, rounding
    )
    result = pd.concat([result, row_inclusion, df_inclusion_labels])

    # Exclusion criteria
    row_exclusion, df_exclusion_labels = process_criteria(
        exclusion_dir, "label", "Exclusion criteria", EXCLUSION_MAPPING, rounding
    )
    result = pd.concat([result, row_exclusion, df_exclusion_labels])

    out_path = f"{output_dir}/study_characteristics.csv"
    result.to_csv(out_path)
    print(f"Study characteristics saved to: {out_path}")


def main(
    output_dir: Path,
    rounding: int,
    summary: bool = True,
    save_preprocessed: bool = True,
):
    df = pd.read_csv(f"{output_dir}/processed_data.csv")
    study_chars(
        df,
        rounding=rounding,
        output_dir=output_dir,
        summary=summary,
        save_preprocessed=save_preprocessed,
    )


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
    rounding = sys.argv[2] if len(sys.argv) > 2 else 2
    summary = "--no-summary" not in sys.argv
    save_preprocessed = "--no-save-preprocessed" not in sys.argv
    main(
        output_dir=output_dir,
        rounding=rounding,
        summary=summary,
        save_preprocessed=save_preprocessed,
    )
