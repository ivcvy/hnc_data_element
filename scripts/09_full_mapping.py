"""Combine all mappings and add reference mappings"""

import pandas as pd
import numpy as np
from mappings import CUSTOM_ENTITY
import json
from pathlib import Path
import sqlalchemy as sa
import sqlalchemy.orm as so
from dotenv import load_dotenv
from omop_alchemy import get_engine_name
from omop_alchemy.cdm.model.vocabulary import ConceptView
from orm_loader.helpers import configure_logging
import sys


DEDUP_COLS = {
    "HeNeCOn": ["obs_description", "target_id"],
    "mCODE": ["obs_description", "target_id"],
    "O3": ["obs_description", "target_id"],
    "OMOP": ["obs_description", "target_description"],
}

EXTRACT_PATTERNS = {
    "domain": r"Domain:\s*([^,]+)",
    "name": r"Name:\s*(.*?)(?=\s*,\s*Domain:)",
    "category": r"Category:\s*(\[[^\]]*\])",
    "measurement": r"Measurement:\s*(\[[^\]]*\])",
    "unit": r"Unit:\s*(\[[^\]]*\])",
}

COLS_TO_KEEP = [
    "domain",
    "name",
    "category",
    "measurement",
    "unit",
    "type",
    "target_id",
    "target_description",
]

MCODE_VOCABULARIES = {"LOINC", "RxNorm", "SNOMED"}
O3_EXCLUDE_VOCAB = {
    "race"
}  # exclude OMOP-only vocabulary, only SNOMED left for O3 references


def load_std(filepath: str, std: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    if {"obs_label", "target_label"}.issubset(df.columns):
        df = df.rename(
            columns={
                "obs_label": "obs_description",
                "target_label": "target_description",
            }
        )
    return df


def process_std(std: str, filepath: str) -> pd.DataFrame:
    df = load_std(filepath, std)
    df["type"] = std

    null_cols = [c for c in ["target_id", "target_description"] if c in df.columns]
    df.loc[df["valid"].isna(), null_cols] = np.nan

    dedup_cols = [c for c in DEDUP_COLS[std] if c in df.columns]
    return df.drop_duplicates(subset=dedup_cols, keep="first")


def extract_fields(df: pd.DataFrame) -> pd.DataFrame:
    for col, pattern in EXTRACT_PATTERNS.items():
        df[col] = df["obs_description"].str.extract(pattern)
    df["domain"] = df["domain"].replace(r"[\[\]']", "", regex=True)
    return df


def apply_custom_domain(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["name"].isin(CUSTOM_ENTITY)
    df.loc[mask, "domain"] = df.loc[mask, "name"].map(CUSTOM_ENTITY)
    return df.replace({"domain": {"infrastructure": "health system"}})


def clean_target_id(x) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def init_session() -> so.Session:
    configure_logging()
    load_dotenv()
    engine = sa.create_engine(get_engine_name(), future=True, echo=False)
    return so.sessionmaker(bind=engine)()


def load_concepts(session: so.Session, target_ids: set) -> pd.DataFrame:
    rows = (
        session.query(ConceptView).filter(ConceptView.concept_id.in_(target_ids)).all()
    )
    concepts = pd.DataFrame([c.to_dict() for c in rows])
    concepts["concept_id"] = concepts["concept_id"].astype(str)
    return concepts


def mcode_reference(df: pd.DataFrame, concepts: pd.DataFrame) -> pd.DataFrame:
    mcode_refs = concepts[concepts["vocabulary_id"].isin(MCODE_VOCABULARIES)]
    matches = df.merge(
        mcode_refs, left_on="target_id", right_on="concept_id", how="inner"
    ).copy()
    matches["type"] = "mCODE"
    matches["target_id"] = (
        matches["vocabulary_id"]
        .str.lower()
        .str.replace("snomed", "snomedct", regex=False)
        + ":"
        + matches["concept_code"]
    )
    matches = matches.drop(
        columns=["concept_id", "concept_name", "concept_code", "vocabulary_id"]
    )
    return pd.concat([df, matches], ignore_index=True)


def find_o3_reference(
    o3_data: dict,
    ref_key: str = "ReferenceSystemForValues",
    numeric_key: str = "NumericCode",
    string_key: str = "StringCode",
) -> dict:
    refs = {}
    if isinstance(o3_data, dict):
        if (
            {ref_key, numeric_key}.issubset(o3_data)
            and o3_data.get(ref_key)
            and o3_data[ref_key].lower() not in {"", "none", "none specified"}
        ):
            refs[o3_data[numeric_key] + f" ({o3_data[string_key]})"] = o3_data[ref_key]
        for v in o3_data.values():
            refs.update(find_o3_reference(v, ref_key, numeric_key, string_key))
    elif isinstance(o3_data, list):
        for item in o3_data:
            refs.update(find_o3_reference(item, ref_key, numeric_key, string_key))
    return refs


def o3_reference(
    df: pd.DataFrame, concepts: pd.DataFrame, o3_data: dict
) -> pd.DataFrame:
    o3_refs = find_o3_reference(o3_data=o3_data)

    names_with_refs = df.loc[df["target_id"].isin(o3_refs), "name"].unique()
    df_candidates = df[
        df["name"].isin(names_with_refs) & df["type"].isin(["OMOP", "O3"])
    ]

    merged = df_candidates.merge(
        concepts, left_on="target_id", right_on="concept_id", how="left"
    )
    merged["ref"] = merged["target_id"].map(o3_refs)

    ref_matches = (
        merged.groupby("name")
        .apply(
            lambda g: any(
                str(vocab) in str(ref)
                for vocab in g["vocabulary_id"].dropna()
                for ref in g["ref"].dropna()
            ),
            include_groups=False,
        )
        .loc[lambda s: s]
        .index.difference(O3_EXCLUDE_VOCAB)
    )

    o3_rows = merged[
        merged["name"].isin(ref_matches)
        & (merged["type"] == "OMOP")
        & (merged["vocabulary_id"] == "SNOMED")
    ].copy()
    o3_rows["type"] = "O3"
    o3_rows["target_id"] = "snomedct:" + o3_rows["concept_code"]
    o3_rows = o3_rows.drop(
        columns=["concept_id", "concept_name", "concept_code", "vocabulary_id", "ref"]
    )

    return pd.concat([df, o3_rows], ignore_index=True)


def filter_valid(df: pd.DataFrame) -> pd.DataFrame:
    has_target = (df["target_id"].notna()) & (df["target_id"] != "")
    group_has_target = has_target.groupby([df["name"], df["type"]]).transform("any")

    return (
        df[~(~has_target & group_has_target)]
        .sort_values(
            by=["valid", "domain", "name", "type", "target_id", "target_description"]
        )
        .reset_index(drop=True)
    )


def main(output_dir: Path, o3_path: Path, reference: bool):
    STD_FILES = {
        "HeNeCOn": f"{output_dir}/combined_henecon_top10_valid.csv",
        "mCODE": f"{output_dir}/combined_mcode_top10_valid.csv",
        "O3": f"{output_dir}/combined_o3_top10_valid_stringcode.csv",
        "OMOP": f"{output_dir}/omop_with_concept_id.csv",
    }

    df_combined = pd.concat(
        [process_std(std, path) for std, path in STD_FILES.items()],
        ignore_index=True,
    )
    df = extract_fields(df_combined)
    df = apply_custom_domain(df)
    df["target_id"] = df["target_id"].apply(clean_target_id)

    if reference:
        target_ids = set(
            df[(df["type"] == "OMOP") & (df["target_id"] != "")]["target_id"]
        )
        session = init_session()
        concepts = load_concepts(session=session, target_ids=target_ids)
        o3_data = json.loads(Path(o3_path).read_text())
        df = mcode_reference(df=df, concepts=concepts)
        df = o3_reference(df=df, concepts=concepts, o3_data=o3_data)
    else:
        df = df[~df["target_description"].str.contains("snomed", case=False, na=False)]

    df = filter_valid(df=df)
    keep = [c for c in COLS_TO_KEEP if c in df.columns]
    df[keep].to_csv(f"{output_dir}/appendix_valid.csv", index=False)


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
    o3_path = sys.argv[2] if len(sys.argv) > 2 else "data/O3_20250128.json"
    reference = "--no-reference" not in sys.argv
    main(output_dir=output_dir, o3_path=o3_path, reference=reference)
