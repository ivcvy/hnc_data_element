"""Find top k similar embeddings in batch mode"""

import os
import glob
import duckdb
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import faiss
import sys


def batch_convert_csv_to_parquet(
    csv_file: str,
    batch_size: int,
    offset: int,
    i: int = 0,
    output_dir: str = "concept_embeddings_batches",
):
    os.makedirs(output_dir, exist_ok=True)

    con = duckdb.connect()

    while True:
        res = con.execute(
            f"""
            SELECT *
            FROM read_csv('{csv_file}', header=True)
            LIMIT {batch_size} OFFSET {offset}
        """
        )

        df = res.fetchdf()

        if df.empty:
            break

        con.execute(
            f"""
            COPY (SELECT * FROM df)
            TO '{output_dir}/concept_embeddings_batch_{i}.parquet'
            (FORMAT 'parquet', COMPRESSION 'zstd');
        """
        )

        offset += batch_size
        i += 1

    con.close()


def convert_embeddings_to_float(
    input_dir: str = "concept_embeddings_batches",
    output_dir: str = "concept_embeddings_float",
):
    os.makedirs(output_dir, exist_ok=True)

    parquets = glob.glob(os.path.join(input_dir, "*.parquet"))

    for infile in parquets:

        base = os.path.basename(infile)
        outfile = os.path.join(output_dir, base)

        duckdb.sql(
            f"""
            COPY (
                SELECT
                    *,
                    list_transform(
                        string_split(
                            replace(replace(embedding::VARCHAR, '[', ''), ']', ''),
                            ','
                        ),
                        x -> CAST(x AS FLOAT)
                    ) AS embedding_vec
                FROM read_parquet('{infile}')
            )
            TO '{outfile}'
            (FORMAT 'parquet', COMPRESSION 'zstd');
        """
        )


def build_faiss_index(
    input_dir: str = "concept_embeddings_float",
    output_dir: str = "concept_embeddings_index",
    embed_col: str = "embedding_vec",
):
    os.makedirs(output_dir, exist_ok=True)

    parquets = glob.glob(os.path.join(input_dir, "*.parquet"))

    for infile in parquets:
        print(f"Reading {infile}")

        table = pq.read_table(infile, columns=[embed_col])
        df = table.to_pandas()

        embeddings = np.vstack(df[embed_col].values).astype("float32")

        dim = embeddings.shape[1]
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        base = os.path.splitext(os.path.basename(infile))[0]
        outfile = os.path.join(output_dir, base + ".index")

        faiss.write_index(index, outfile)


def top_k_search(
    obs_index_file: str = "text_embedding_3_large_obs_embeddings.index",
    target_dir: str = "concept_embeddings_index",
    output_dir: str = "omop_topk",
    k: int = 10,
):
    os.makedirs(output_dir, exist_ok=True)

    obs_index = faiss.read_index(obs_index_file)
    obs_vectors = obs_index.reconstruct_n(0, obs_index.ntotal)
    faiss.normalize_L2(obs_vectors)

    index_files = glob.glob(os.path.join(target_dir, "*.index"))

    for infile in index_files:
        base = os.path.splitext(os.path.basename(infile))[0]
        outfile = os.path.join(output_dir, f"{base}_top10.csv")

        target_index = faiss.read_index(infile)
        target_vectors = target_index.reconstruct_n(0, target_index.ntotal)
        faiss.normalize_L2(target_vectors)

        dim = target_vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(target_vectors)

        scores, indices = index.search(obs_vectors, k)

        df = pd.DataFrame(
            {
                "obs_index": np.repeat(np.arange(obs_vectors.shape[0]), k),
                "target_index": indices.flatten(),
                "score": scores.flatten(),
                "filename": base,
            }
        )
        df.to_csv(outfile, index=False)


def combine_topk(
    input_dir: str = "omop_topk/",
    k: int = 10,
    output_file: str = "omop_topk_merged.csv",
):
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)

    full_topk = (
        full.sort_values(["obs_index", "score"], ascending=[True, False])
        .groupby("obs_index")
        .head(k)
        .reset_index(drop=True)
    )

    full_topk.to_csv(output_file, index=False)


def label_topk_with_concept_names(
    csv_file: str = "omop_topk_merged.csv",
    parquet_dir: str = "concept_embeddings_float",
    obs_file: str = "text_embedding_3_large_obs_embeddings.csv",
    output_file: str = "omop_topk_labels.csv",
    label_col: str = "label",
):
    con = duckdb.connect()

    df = con.execute(f"SELECT * FROM read_csv_auto('{csv_file}')").df()
    obs_df = con.execute(f"SELECT * FROM read_csv_auto('{obs_file}')").df()

    df["filename"] = df["filename"].apply(os.path.basename)

    def get_obs_label(obs_index):
        if obs_index < 0 or obs_index >= len(obs_df):
            return None
        return obs_df.iloc[obs_index]["text"]

    df["obs_label"] = df["obs_index"].apply(get_obs_label)

    def get_label_from_parquet(filename, target_index):
        parquet_path = os.path.join(parquet_dir, f"{filename}.parquet")
        if not os.path.exists(parquet_path):
            return None

        query = f"""
            SELECT {label_col}
            FROM read_parquet('{parquet_path}')
            LIMIT 1
            OFFSET {target_index}
        """
        result = con.execute(query).fetchone()
        if result is None:
            return None
        return result[0]

    df["target_label"] = df.apply(
        lambda row: get_label_from_parquet(row["filename"], row["target_index"]), axis=1
    )

    df.to_csv(output_file, index=False)


def main(csv_file: str, batch_size: int, offset: int, k: int):
    batch_convert_csv_to_parquet(
        csv_file=csv_file, batch_size=batch_size, offset=offset
    )
    convert_embeddings_to_float()
    build_faiss_index()
    top_k_search(k=k)
    combine_topk(k=k)
    label_topk_with_concept_names()


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "data/concept_embeddings.csv"
    batch_size = sys.argv[2] if len(sys.argv) > 2 else 100_000
    offset = sys.argv[3] if len(sys.argv) > 3 else 0
    k = sys.argv[4] if len(sys.argv) > 4 else 10
    main(csv_file=csv_file, batch_size=batch_size, offset=offset, k=k)
