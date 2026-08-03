import argparse
import os

import pandas as pd


def make_cell_batch(dataframe):
    extracted = dataframe["cell_type"].astype(str).str.extract(
        r"^\s*([^()]+)\s*(?:\(([^()]*)\))?\s*$"
    )
    dataframe["cell_type"] = extracted[0].str.strip()
    extracted_batch = extracted[1].fillna("").str.strip()
    if "cell_batch" in dataframe.columns:
        existing_batch = dataframe["cell_batch"].fillna("").astype(str).str.strip()
        dataframe["cell_batch"] = extracted_batch.where(extracted_batch != "", existing_batch)
    else:
        dataframe["cell_batch"] = extracted_batch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare two proteomics datasets and create a merged expression matrix."
    )
    parser.add_argument("--x1_data", required=True, help="Raw expression CSV for dataset 1; cells may be rows or columns.")
    parser.add_argument("--x1_cell", required=True, help="Cell annotation CSV for dataset 1.")
    parser.add_argument("--x2_data", required=True, help="Raw expression CSV for dataset 2; cells may be rows or columns.")
    parser.add_argument("--x2_cell", required=True, help="Cell annotation CSV for dataset 2.")
    parser.add_argument("--x1_name", default="N2", help="Dataset 1 name used in output file names.")
    parser.add_argument("--x2_name", default="nanoPOTS", help="Dataset 2 name used in output file names.")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory; created automatically if needed.")
    return parser.parse_args()


def load_expression(path, annotation_index):
    expression = pd.read_csv(path, index_col=0)
    expression.index = expression.index.astype(str)
    expression.columns = expression.columns.astype(str)
    annotation_ids = set(annotation_index.astype(str))
    row_matches = len(annotation_ids.intersection(expression.index))
    column_matches = len(annotation_ids.intersection(expression.columns))
    if row_matches > column_matches:
        expression = expression.T
    return expression.fillna(0.0)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    x1_cell = pd.read_csv(args.x1_cell, index_col=0)
    x2_cell = pd.read_csv(args.x2_cell, index_col=0)
    x1_data = load_expression(args.x1_data, x1_cell.index)
    x2_data = load_expression(args.x2_data, x2_cell.index)

    if "cell_type" not in x1_cell.columns or "cell_type" not in x2_cell.columns:
        raise ValueError("Both cell annotation files must contain a cell_type column.")

    make_cell_batch(x1_cell)
    make_cell_batch(x2_cell)

    merged = pd.concat([x1_data, x2_data], axis=1).fillna(0.0)

    x1_data.to_csv(os.path.join(args.output_dir, f"{args.x1_name}_data.csv"))
    x1_cell.to_csv(os.path.join(args.output_dir, f"{args.x1_name}_cell.csv"))
    x2_data.to_csv(os.path.join(args.output_dir, f"{args.x2_name}_data.csv"))
    x2_cell.to_csv(os.path.join(args.output_dir, f"{args.x2_name}_cell.csv"))
    merged.to_csv(os.path.join(args.output_dir, "merged_data.csv"))

    print("Saved prepared data files to:", os.path.abspath(args.output_dir))


if __name__ == "__main__":
    main()
