import argparse
import os
import time
from typing import Dict, List, Tuple

import pandas as pd
import requests


BACKOFF_BASE = 1.5


def robust_post(url, data, max_retries=5, connect_timeout=10, read_timeout=60):
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                data=data,
                timeout=(connect_timeout, read_timeout),
            )
            response.raise_for_status()
            return response
        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.HTTPError,
        ) as error:
            last_error = error
            sleep_seconds = (BACKOFF_BASE ** attempt) + (0.1 * attempt)
            time.sleep(sleep_seconds)
    raise last_error


def map_to_string_ids_batch(
    ids: List[str],
    species: int,
    batch_size: int,
    max_retries: int,
    connect_timeout: int,
    read_timeout: int,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    url = "https://string-db.org/api/tsv/get_string_ids"
    original_to_string = {}
    string_to_original = {}
    ids = [str(value) for value in ids]

    for start in range(0, len(ids), batch_size):
        chunk = ids[start:start + batch_size]
        try:
            response = robust_post(
                url,
                data={
                    "identifiers": "\n".join(chunk),
                    "species": species,
                    "limit": 1,
                    "echo_query": 1,
                },
                max_retries=max_retries,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )
        except Exception as error:
            print(f"[WARNING] get_string_ids batch {start}-{start + len(chunk) - 1} failed: {error}")
            continue

        lines = response.text.strip().splitlines()
        if not lines:
            continue
        header = lines[0].split("\t")
        column_index = {name: index for index, name in enumerate(header)}
        query_index = column_index.get("queryItem")
        string_index = column_index.get("stringId")
        if query_index is None or string_index is None:
            print("[WARNING] Unexpected get_string_ids response header; batch skipped.")
            continue

        for line in lines[1:]:
            columns = line.split("\t")
            if len(columns) <= max(query_index, string_index):
                continue
            original_id = columns[query_index]
            string_id = columns[string_index]
            original_to_string.setdefault(original_id, string_id)
            string_to_original.setdefault(string_id, original_id)

    return original_to_string, string_to_original


def get_interactions_batch(
    string_ids: List[str],
    species: int,
    required_score: int,
    batch_size: int,
    max_retries: int,
    connect_timeout: int,
    read_timeout: int,
) -> pd.DataFrame:
    url = "https://string-db.org/api/tsv/network"
    all_rows = []
    unique_ids = list(dict.fromkeys(string_ids))

    for start in range(0, len(unique_ids), batch_size):
        chunk = unique_ids[start:start + batch_size]
        try:
            response = robust_post(
                url,
                data={
                    "identifiers": "%0d".join(chunk),
                    "species": species,
                    "required_score": required_score,
                    "caller_identity": "scPPInt_ppi_network",
                },
                max_retries=max_retries,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )
        except Exception as error:
            print(f"[WARNING] network batch {start}-{start + len(chunk) - 1} failed: {error}")
            continue

        lines = response.text.strip().splitlines()
        if not lines:
            continue
        header = lines[0].split("\t")
        try:
            protein_a_index = header.index("stringId_A")
            protein_b_index = header.index("stringId_B")
            score_index = header.index("score")
        except ValueError:
            print("[WARNING] Required network response columns are missing; batch skipped.")
            continue

        chunk_set = set(chunk)
        for line in lines[1:]:
            columns = line.split("\t")
            if len(columns) <= max(protein_a_index, protein_b_index, score_index):
                continue
            protein_a = columns[protein_a_index]
            protein_b = columns[protein_b_index]
            if protein_a in chunk_set and protein_b in chunk_set and protein_a != protein_b:
                try:
                    score = float(columns[score_index])
                except ValueError:
                    continue
                all_rows.append((protein_a, protein_b, score))

    if not all_rows:
        return pd.DataFrame(columns=["protein1", "protein2", "score"])
    return pd.DataFrame(all_rows, columns=["protein1", "protein2", "score"]).drop_duplicates()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrieve a STRING PPI network for proteins in an expression matrix."
    )
    parser.add_argument("--merged_data", required=True, help="Merged expression CSV whose row index contains the protein identifiers.")
    parser.add_argument("--species", required=True, type=int, help="NCBI taxonomy ID, for example 10090 for mouse or 9606 for human.")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory; created automatically if needed.")
    parser.add_argument("--output_name", default="ppi_edges_protein_ids.csv", help="Output PPI edge CSV name.")
    parser.add_argument("--score_threshold", type=int, default=700, help="STRING required score from 0 to 1000.")
    parser.add_argument("--batch_size_map", type=int, default=100)
    parser.add_argument("--batch_size_net", type=int, default=200)
    parser.add_argument("--connect_timeout", type=int, default=10)
    parser.add_argument("--read_timeout", type=int, default=60)
    parser.add_argument("--max_retries", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.score_threshold <= 1000:
        raise ValueError("score_threshold must be between 0 and 1000.")

    proteins = list(pd.read_csv(args.merged_data, index_col=0).index.astype(str))
    original_to_string, string_to_original = map_to_string_ids_batch(
        proteins,
        species=args.species,
        batch_size=args.batch_size_map,
        max_retries=args.max_retries,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
    )
    mapped_ids = [original_to_string[protein] for protein in proteins if protein in original_to_string]
    if not mapped_ids:
        raise SystemExit(
            f"No protein IDs were mapped to STRING for species {args.species}. Check the species ID and input identifiers."
        )

    missing = [protein for protein in proteins if protein not in original_to_string]
    if missing:
        print(f"[INFO] {len(missing)} input proteins were not mapped and will be ignored. Examples: {missing[:3]}")

    edges = get_interactions_batch(
        mapped_ids,
        species=args.species,
        required_score=args.score_threshold,
        batch_size=args.batch_size_net,
        max_retries=args.max_retries,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
    )

    if not edges.empty:
        edges = edges.copy()
        edges["protein1"] = edges["protein1"].map(lambda value: string_to_original.get(value, value))
        edges["protein2"] = edges["protein2"].map(lambda value: string_to_original.get(value, value))
        normalized_pairs = [tuple(sorted((protein_a, protein_b))) for protein_a, protein_b in zip(edges["protein1"], edges["protein2"])]
        edges["u_protein1"] = [pair[0] for pair in normalized_pairs]
        edges["u_protein2"] = [pair[1] for pair in normalized_pairs]
        edges = edges.groupby(["u_protein1", "u_protein2"], as_index=False)["score"].max()
        edges = edges.rename(columns={"u_protein1": "protein1", "u_protein2": "protein2"})

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    edges.to_csv(output_path, index=False)
    print(
        f"Saved PPI edge table to {output_path}. Species={args.species}; "
        f"score threshold={args.score_threshold}; edges={len(edges)}."
    )


if __name__ == "__main__":
    main()
