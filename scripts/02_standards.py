"""Encode sentence and search top-k similarity in target standards"""

from pathlib import Path
import os
import pandas as pd
import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import sys


def load_obs(output_dir: str) -> list:
    print("Loading observed data...")
    df = pd.read_csv(f"{output_dir}/processed_data.csv")
    result = df.groupby("name").agg(
        {
            "entity_name": lambda x: list(x.dropna().unique()),
            "category": lambda x: list(x.dropna().unique()),
            "measurement": lambda x: list(x.dropna().unique()),
            "unit": lambda x: list(x.dropna().unique()),
        }
    )

    obs_sentences = []
    for row in result.itertuples():
        obs_sentences.append(
            f"Name: {row.Index}, Domain: {row.entity_name}, Category: {row.category}, Measurement: {row.measurement}, Unit: {row.unit}"
        )

    obs_df = pd.DataFrame(obs_sentences, columns=["obs_label"])
    obs_df.to_csv(f"{output_dir}/obs_label.csv", index=False)

    return obs_sentences


def load_o3(input_dir: str, output_dir: str) -> dict:
    print("Loading O3 data...")
    with open(f"{input_dir}/O3_20250128.json", "r") as json_file:
        o3 = json.load(json_file)
        o3_dict = {}

        for key_element in o3:
            for attr in key_element.get("list_attributes", []):
                numeric_code = attr.get("NumericCode", "")
                name = attr.get("ValueName", "")
                definition = attr.get("Definition", "")
                data_type = attr.get("ValueDataType", "")
                standard_value = attr.get("StandardValuesList", [])

                o3_attr = f"Name: {name}, Definition: {definition}, Data type: {data_type}, Standard values: {standard_value}"

                if o3_attr and numeric_code:
                    o3_dict[numeric_code] = o3_attr

        o3_df = pd.DataFrame(list(o3_dict.items()), columns=["label_id", "label"])
        o3_df.to_csv(f"{output_dir}/o3_label.csv", index=False)

    return o3_dict


def load_mcode(input_dir: str, output_dir: str) -> dict:
    print("Loading mCODE data...")
    with open(f"{input_dir}/mCODEDataDictionary-STU4.json", "r") as json_file:
        mcode = json.load(json_file)
        mcode_dict = {}

        for _, e in mcode.items():
            for attr in e if isinstance(e, list) else []:
                profile_title = attr.get("Profile Title", "")
                data_element_name = attr.get("Data Element Name", "")
                uri = attr.get("Value Set URI", "")
                r4 = attr.get("FHIR Element (R4)", "")
                value_set_uri = attr.get("Value set URI", "")
                url = attr.get("url", "")
                code_system = attr.get("Code system", "")
                code = attr.get("Code", "")

                if uri or data_element_name:
                    string_code = f"{profile_title}_{data_element_name}_{uri}_{r4}"
                elif url:
                    string_code = url
                elif code:
                    string_code = f"{value_set_uri}_{code_system}_{code}"
                else:
                    continue

                if profile_title and data_element_name:
                    name = f"{profile_title}_{data_element_name}"
                elif attr.get("title"):
                    name = attr.get("title")
                elif code_system and code:
                    name = f"{code_system}_{code}"
                else:
                    continue

                definition = (
                    attr.get("Definition", "")
                    or attr.get("description", "")
                    or attr.get("Code description", "")
                )
                data_type = attr.get("Data Type", "")

                mcode_attr = (
                    f"Name: {name}, Definition: {definition}, Data type: {data_type}"
                )

                if string_code and mcode_attr:
                    mcode_dict[string_code] = mcode_attr

        mcode_df = pd.DataFrame(list(mcode_dict.items()), columns=["label_id", "label"])
        mcode_df.to_csv(f"{output_dir}/mcode_label.csv", index=False)

        return mcode_dict


def load_henecon(input_dir: str, output_dir: str) -> dict:
    print("Loading HENECON data...")
    henecon = pd.read_csv(f"{input_dir}/HENECON.csv")
    henecon_dict = {
        row[
            "Class ID"
        ]: f"Name: {row['Preferred Label']}, Definition: {row['Definitions']}"
        for _, row in henecon.iterrows()
        if row["Class ID"] and row["Preferred Label"]
    }
    henecon_df = pd.DataFrame(list(henecon_dict.items()), columns=["label_id", "label"])
    henecon_df.to_csv(f"{output_dir}/henecon_label.csv", index=False)

    return henecon_dict


def safe_filename(model: str) -> str:
    return model.replace("/", "_").replace("-", "_")


def sentences_to_index(
    output_dir: str,
    model: str,
    sentences: list,
    name: str,
    device: str = "cpu",
    batch_size=16,
):
    model_name = safe_filename(model)

    print(f"Encoding sentences from {name} using model {model}...")
    encoder = SentenceTransformer(model, device=device)
    embeddings = encoder.encode(
        sentences,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, f"{output_dir}/{model_name}_{name}_embeddings.index")


def sentences_to_index_openai(
    output_dir: str, model: str, sentences: list, name: str, batch_size: int = 16
):
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file.")

    client = OpenAI(api_key=api_key)

    model_name = safe_filename(model)

    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch, dimensions=1024)
        batch_embs = [d.embedding for d in response.data]
        embeddings.extend(batch_embs)

    embeddings = np.array(embeddings, dtype="float32")
    dim = embeddings.shape[1]
    print(f"Generated embeddings: {embeddings.shape[0]} x {dim}")

    csv_path = f"{output_dir}/{model_name}_{name}_embeddings.csv"
    embedding_strings = [emb.tolist() for emb in embeddings]
    df = pd.DataFrame({"text": sentences, "embedding": embedding_strings})
    df.to_csv(csv_path, index=False)
    print(f"Saved embeddings CSV to {csv_path}")

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    out_path = f"{output_dir}/{model_name}_{name}_embeddings.index"
    faiss.write_index(index, out_path)
    print(f"FAISS index saved to {out_path}")


def top_k_search(output_dir: str, model: str, name: str, k: int):
    model_name = safe_filename(model)

    print(f"Performing top-{k} search from obs to {name} using model {model}...")
    obs_index = faiss.read_index(f"{output_dir}/{model_name}_obs_embeddings.index")
    target_index = faiss.read_index(
        f"{output_dir}/{model_name}_{name}_embeddings.index"
    )
    obs_vectors = obs_index.reconstruct_n(0, obs_index.ntotal)

    scores, indices = target_index.search(obs_vectors, k)

    obs_label = pd.read_csv(f"{output_dir}/obs_label.csv")
    if name == "obs":
        target_label = obs_label
    else:
        target_label = pd.read_csv(f"{output_dir}/{name}_label.csv")
    results = []
    for i, (score_row, idx_row) in enumerate(zip(scores, indices)):
        obs_description = obs_label.iloc[i, 0]

        for j in range(k):
            if name == "obs":
                target_id = None
                target_description = obs_label.iloc[idx_row[j], 0]
            else:
                target_id = target_label.iloc[idx_row[j], 0]
                target_description = target_label.iloc[idx_row[j], 1]

            results.append(
                {
                    "obs_description": obs_description,
                    "target_id": target_id,
                    "target_description": target_description,
                    "score": float(score_row[j]),
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/{model_name}_{name}_top{k}.csv", index=False)

    return scores, indices


def main(input_dir: Path, output_dir: Path):
    models = [
        "abhinand/MedEmbed-large-v0.1",
        "Salesforce/SFR-Embedding-Mistral",
    ]

    standards = {
        "obs": lambda: load_obs(output_dir=output_dir),
        "o3": lambda: list(load_o3(input_dir, output_dir).values()),
        "mcode": lambda: list(load_mcode(input_dir, output_dir).values()),
        "henecon": lambda: list(load_henecon(input_dir, output_dir).values()),
    }

    print("Loading datasets...")
    sentence_sets = {name: loader() for name, loader in standards.items()}

    print("Building embedding indices...")
    for model in models:
        for name, sentences in sentence_sets.items():
            print(f"Indexing {name} with {model}")
            sentences_to_index(
                output_dir=output_dir,
                model=model,
                sentences=sentences,
                name=name,
            )

    print("Running top-k searches...")
    for model in models:
        for name in sentence_sets:
            if name == "obs":
                continue
            top_k_search(
                output_dir=output_dir,
                model=model,
                name=name,
                k=10,
            )

    print("Building OpenAI embeddings for observed data element...")
    sentences_to_index_openai(
        output_dir=output_dir,
        model="text-embedding-3-large",
        sentences=sentence_sets["obs"],
        name="obs",
        batch_size=16,
    )


if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    main(input_dir=input_dir, output_dir=output_dir)
