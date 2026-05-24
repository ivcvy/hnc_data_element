"""Generate SSSOM/TSV and CSV of mappings"""

import csv
import re
import sys
from datetime import date
from sssom.parsers import parse_sssom_table
from sssom.validators import validate
from sssom.util import MappingSetDataFrame
import io
from contextlib import redirect_stderr


OSF_DOI = "10.17605/OSF.IO/ZKAVC"
STUDY_BASE_IRI = f"https://doi.org/{OSF_DOI}#"
MAPPING_SET_ID = f"https://doi.org/{OSF_DOI}"
MAPPING_SET_TITLE = "Observational Study Data Elements to Target Ontology Mappings"
MAPPING_SET_VERSION = "1.0.0"
MAPPING_DATE = str(date.today())
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
OBJECT_SOURCES = {
    "HeNeCOn": "https://bioportal.bioontology.org/ontologies/HENECON",
    "O3": "https://aapmbdsc.azurewebsites.net/Upload/O3_20250128.json",
    "OMOP": "https://athena.ohdsi.org/search-terms/terms/",
    "mCODE": "https://github.com/HL7/fhir-mCODE-ig/blob/master/data-dictionary/mCODEDataDictionary-STU4.json",
}
CURIE_MAP = {
    "study": STUDY_BASE_IRI,
    "henecon": "http://ontology.lst.tfo.upm.es/BD2D#",
    "henecon_chemo": "http://ontology.lst.tfo.upm.es/BD2D/chemo#",
    "henecon_clinical": "http://ontology.lst.tfo.upm.es/BD2D/clinical#",
    "henecon_ctn": "http://ontology.lst.tfo.upm.es/BD2D/ctn#",
    "henecon_follow": "http://ontology.lst.tfo.upm.es/BD2D/follow#",
    "henecon_img": "http://ontology.lst.tfo.upm.es/BD2D/img#",
    "henecon_patho": "http://ontology.lst.tfo.upm.es/BD2D/patho#",
    "henecon_patient": "http://ontology.lst.tfo.upm.es/BD2D/patient#",
    "henecon_hrs": "http://ontology.lst.tfo.upm.es/BD2D/population/HRS#",
    "henecon_hrschemo": "http://ontology.lst.tfo.upm.es/BD2D/population/HRSchemo#",
    "henecon_hrsclinical": "http://ontology.lst.tfo.upm.es/BD2D/population/HRSclinical#",
    "henecon_hrsctn": "http://ontology.lst.tfo.upm.es/BD2D/population/HRSctn#",
    "henecon_hrspatho": "http://ontology.lst.tfo.upm.es/BD2D/population/HRSpatho#",
    "henecon_hrsradio": "http://ontology.lst.tfo.upm.es/BD2D/population/HRSradio#",
    "henecon_hrssurge": "http://ontology.lst.tfo.upm.es/BD2D/population/HRSsurge#",
    "henecon_ps": "http://ontology.lst.tfo.upm.es/BD2D/population/PS#",
    "henecon_psclinical": "http://ontology.lst.tfo.upm.es/BD2D/population/PSclinical#",
    "henecon_psctn": "http://ontology.lst.tfo.upm.es/BD2D/population/PSctn#",
    "henecon_pspatho": "http://ontology.lst.tfo.upm.es/BD2D/population/PSpatho#",
    "henecon_pst1": "http://ontology.lst.tfo.upm.es/BD2D/population/PSt1#",
    "henecon_pst2": "http://ontology.lst.tfo.upm.es/BD2D/population/PSt2#",
    "henecon_pst3": "http://ontology.lst.tfo.upm.es/BD2D/population/PSt3#",
    "henecon_pst4": "http://ontology.lst.tfo.upm.es/BD2D/population/PSt4#",
    "henecon_pst5": "http://ontology.lst.tfo.upm.es/BD2D/population/PSt5#",
    "henecon_qol": "http://ontology.lst.tfo.upm.es/BD2D/qol#",
    "henecon_radio": "http://ontology.lst.tfo.upm.es/BD2D/radio#",
    "henecon_risk": "http://ontology.lst.tfo.upm.es/BD2D/risk#",
    "henecon_surge": "http://ontology.lst.tfo.upm.es/BD2D/surge#",
    "henecon_tox": "http://ontology.lst.tfo.upm.es/BD2D/tox#",
    "neomark": "http://neomark.owl#",
    "Ontology1325521724189": "http://www.semanticweb.org/ontologies/2012/0/Ontology1325521724189.owl#",
    "o3": "http://example.org/O3_20250128.owl#",
    "omop": "https://athena.ohdsi.org/search-terms/terms/",
    "mcode_r4": "https://hl7.org/fhir/R4/",
    "mcode": "http://hl7.org/fhir/us/mcode/StructureDefinition/",
    "mcode_valueset": "http://hl7.org/fhir/us/mcode/ValueSet/",
    "snomedct": "http://snomed.info/id/",
    "loinc": "http://loinc.org/",
    "rxnorm": "http://purl.bioontology.org/ontology/RXNORM/",
    "semapv": "https://w3id.org/semapv/vocab/",
    "skos": "http://www.w3.org/2004/02/skos/core#",
}


def to_local_id(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s


def make_subject_id(name: str) -> str:
    return f"study:{to_local_id(name)}"


def add_predicate() -> str:
    return "skos:relatedMatch"


def normalise_target_id(raw: str) -> str:
    raw = raw.strip()

    for prefix, base in CURIE_MAP.items():
        if raw.startswith(base) and "http://snomed.info/sct" not in raw:
            local = raw[len(base) :]
            return f"{prefix}:{local}"

    snomed_sct = re.search(r"http://snomed\.info/sct_(\d+)", raw)
    if snomed_sct:
        return f"snomedct:{snomed_sct.group(1)}"

    if raw.startswith("O3_"):
        local = re.search(r"\((.+?)\)", raw)
        if local:
            return f"o3:{local.group(1)}"

    if raw.isdigit():
        return f"omop:{raw}"

    fhir_path_match = re.search(r"_([A-Z][a-zA-Z]+\.[a-zA-Z].+)$", raw)
    if fhir_path_match:
        fhir_path = fhir_path_match.group(1)
        fhir_path = re.sub(r"\[(.+?)\]", lambda m: f"%5B{m.group(1)}%5D", fhir_path)
        fhir_path = fhir_path.replace(":", "%3A")
        return f"mcode_r4:{fhir_path}"

    return raw


def build_comment(row: dict) -> str:
    parts = [
        f"domain={row['domain']}",
        f"category={row['category']}",
        f"measurement={row['measurement']}",
    ]
    if row.get("unit"):
        parts.append(f"unit={row['unit']}")
    return "; ".join(parts)


def sssom_header() -> str:
    object_source_comment = "\n".join(
        f"#     {k}: {v}" for k, v in OBJECT_SOURCES.items()
    )
    lines = [
        f"# mapping_set_id: {MAPPING_SET_ID}",
        f"# mapping_set_title: {MAPPING_SET_TITLE}",
        f"# mapping_set_version: {MAPPING_SET_VERSION}",
        f"# mapping_date: {MAPPING_DATE}",
        f"# license: {LICENSE}",
        "# object_source: multiple",
        "# comment: |",
        "#   object sources:",
        object_source_comment,
        "#   mcode_r4 prefix links to https://hl7.org/fhir/R4/Resource.html#Resource.element for Resource.element local part",
        "# curie_map:",
    ]
    for prefix, uri in CURIE_MAP.items():
        lines.append(f'#   {prefix}: "{uri}"')
    return "\n".join(lines)


SSSOM_COLS = [
    "subject_id",
    "subject_label",
    "predicate_id",
    "object_id",
    "mapping_justification",
    "comment",
]


def convert(input_path: str, sssom_path: str) -> None:
    rows = []

    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target_raw = row.get("target_id", "").strip()
            if not target_raw:
                continue

            object_id = normalise_target_id(target_raw)
            source_type = row.get("type", "")

            sssom_row = {
                "subject_id": make_subject_id(row["name"]),
                "subject_label": row["name"].strip(),
                "predicate_id": add_predicate(),
                "object_id": object_id,
                "mapping_justification": "semapv:ManualMappingCuration",
                "comment": build_comment(row),
            }
            rows.append(sssom_row)

    with open(sssom_path, "w", newline="", encoding="utf-8") as f:
        f.write(sssom_header() + "\n")
        writer = csv.DictWriter(f, fieldnames=SSSOM_COLS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} SSSOM to {sssom_path}.")


def validate_sssom_tsv(filepath: str) -> None:
    stderr_capture = io.StringIO()
    with redirect_stderr(stderr_capture):
        msdf: MappingSetDataFrame = parse_sssom_table(filepath)
        validate(msdf)

    stderr_output = stderr_capture.getvalue()

    if "Error:" in stderr_output:
        print(stderr_output, file=sys.stderr)
        raise ValueError(f"{filepath} is not a valid SSSOM/TSV.")

    print(f"{filepath} is a valid SSSOM/TSV.", file=sys.stderr)


def main(input_csv: str, output_tsv: str):
    convert(input_csv, output_tsv)
    validate_sssom_tsv(output_tsv)


if __name__ == "__main__":
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "output/appendix_valid.csv"
    output_tsv = sys.argv[2] if len(sys.argv) > 2 else "output/mapping.sssom.tsv"
    main(input_csv=input_csv, output_tsv=output_tsv)
