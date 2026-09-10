import pandas as pd
import argparse
import os

def parque2csv(meta_path: str, csv_path: str = None, overwrite: bool = False):
    """Convert a parquet file to CSV, keeping the header and all columns.

    When the parquet stores its row labels as data — the dataset-native
    layout where subtask/task descriptions live in the row index, mirroring
    tasks.parquet — those labels are written as the trailing CSV column
    with the index name as its header, matching the physical column order
    of the file (index columns are stored last), so nothing is dropped on
    the way out.

    Args:
        meta_path (str): Path to the input parquet file.
        csv_path (str, optional): Path to the csv file. If not provided, it
            will be derived from the input path.
    """
    assert meta_path.endswith(".parquet"), "Input file must be a parquet file."
    if csv_path is None:
        csv_path = meta_path.replace(".parquet", ".csv")
    if os.path.exists(csv_path) and not overwrite:
        print(f"{csv_path} already exists. Stop writing to avoid accidently delete data. Please pass --overwrite to enforce overwrite.")
        return

    df = pd.read_parquet(meta_path)
    # A non-default index carries data (e.g. the subtask descriptions).
    # to_csv would print the index as the leftmost column, but the parquet
    # stores it last (data columns first) — promote it to a column and move
    # it to the end so the CSV mirrors the file's physical column order.
    if not isinstance(df.index, pd.RangeIndex):
        idx_name = df.index.name or "index"
        df = df.reset_index()
        df = df[[c for c in df.columns if c != idx_name] + [idx_name]]
    df.to_csv(csv_path, index=False)

def csv2parque(csv_path: str, target_parquet: str = None):
    """Convert a CSV file to parquet, restoring the dataset-native layout.

    The subtask/task description column — ``subtask`` or ``task`` when
    present, otherwise the lone column not matching the ``*_index`` /
    ``*_id`` numeric convention — becomes the parquet row index (mirroring
    tasks.parquet). All remaining columns are written as-is, so extra
    metadata columns edited into the CSV (e.g. manipulator / object /
    destination annotations) are preserved. The header is always kept.

    Args:
        csv_path (str): Path to the input CSV file.
        target_parquet (str, optional): Path to the parquet file. If not
            provided, it will be derived from the input path.
    """
    df = pd.read_csv(csv_path)

    # Normalize headers edited by hand/spreadsheets: trim stray whitespace
    # and drop unlabelled trailing columns pandas names 'Unnamed: N' (they
    # only appear when a ragged extra cell follows the real columns).
    df.columns = [str(c).strip() for c in df.columns]
    df = df[[c for c in df.columns if not c.startswith("Unnamed:")]]

    # Restore the dataset-native layout: the subtask/task description
    # column carries the row labels of the written parquet, even when the
    # CSV also carries extra descriptive metadata columns.
    key_cols = [
        c for c in df.columns
        if not (c.endswith("_index") or c.endswith("_id"))
        and not pd.api.types.is_integer_dtype(df[c])
    ]
    key = next((c for c in ("subtask", "task") if c in key_cols), None)
    if key is None and len(key_cols) == 1:
        key = key_cols[0]
    keep_index = key is not None
    if keep_index:
        df = df.set_index(key)

    # Automatically cast types
    for col in df.columns:
        # Rule 1: Numeric index columns become int64
        if col.endswith("_index") or col.endswith("_id") or pd.api.types.is_integer_dtype(df[col]):
            # fillna(0) prevents pandas from converting integer columns to float if an empty cell exists
            df[col] = df[col].fillna(0).astype("int64")
        # Rule 2: All other descriptive metadata fields become string
        else:
            df[col] = df[col].fillna("").astype("string")

    # Write out to parquet, keeping the row index (the descriptions) when
    # one was restored from the CSV.
    if target_parquet is None:
        target_parquet = csv_path.replace(".csv", ".parquet")
    df.to_parquet(target_parquet, index=keep_index)

    # Display resolved types for verification
    print("Resolved Schema:")
    print(df.dtypes)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert between parquet and CSV formats.")
    parser.add_argument("parque_path", type=str, help="Path to the input file parquet.")
    parser.add_argument("--csv-path", type=str, default=None, help="Path to the csv file. If not provided, will be derived from input path.")
    parser.add_argument("--to-csv", action="store_true", help="Convert from parquet to CSV.")
    parser.add_argument("--to-parquet", action="store_true", help="Convert from CSV to parquet.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files if they exist.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.to_csv:
        parque2csv(args.parque_path, args.csv_path, overwrite=args.overwrite)
    elif args.to_parquet:
        csv2parque(args.parque_path, args.csv_path)
    else:
        raise ValueError("Please specify either --to_csv or --to_parquet.")