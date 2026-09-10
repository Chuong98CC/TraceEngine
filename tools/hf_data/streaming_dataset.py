import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset

# Always downloaded: the dataset metadata (meta/* also covers meta/*.jsonl).
# data/ (parquet) and videos/ are added on top unless the matching --ignore-* flag is given.
ALWAYS_ALLOW_PATTERNS = ["meta/*", "*.json"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a LeRobotDataset's metadata + tabular data and "
                    "open it as a StreamingLeRobotDataset."
    )
    parser.add_argument("repo_id_arg", nargs="?", default=None, metavar="repo_id",
                        help="dataset repo id as seen on the Hub, "
                             "e.g. Kronze157/astribot_making_coffee_vlva_full")
    parser.add_argument("--repo-id", "-id", dest="repo_id_opt", default=None,
                        metavar="REPO_ID",
                        help="same as the positional repo_id")
    parser.add_argument("--local-dir", "-d", default=None,
                        help="directory to download the metadata/parquet into; "
                             "default: /data/<repo name>")
    parser.add_argument("--ignore-videos", action="store_true",
                        help="skip the videos/ tree — meta, *.json and data only")
    parser.add_argument("--ignore-data", action="store_true",
                        help="skip the data/ parquet tree — meta and *.json only; the "
                             "dataset cannot be read locally without it")
    parser.add_argument("--repo-type", default="dataset",
                        choices=("dataset", "bucket"),
                        help="Hub repo type (default: %(default)s)")
    parser.add_argument("--buffer-size", type=int, default=1000,
                        help="shuffle buffer size when streaming "
                             "(default: %(default)s)")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="iterate the dataset in order instead of shuffling")

    args = parser.parse_args()
    if args.repo_id_arg and args.repo_id_opt and args.repo_id_arg != args.repo_id_opt:
        parser.error(f"conflicting repo ids: {args.repo_id_arg!r} (positional) vs "
                     f"{args.repo_id_opt!r} (--repo-id)")
    args.repo_id = args.repo_id_arg or args.repo_id_opt
    if args.repo_id is None:
        parser.error("a repo id is required, e.g. "
                     "`streaming_dataset.py Kronze157/astribot_making_coffee_vlva_full`")
    return args


def main():
    args = parse_args()

    # Load environment variables from .env file into os.environ (before the token is read)
    # create .env file in the repo and paste your HF_TOKEN="your-hf-token-to-access-data"
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    allow_patterns = list(ALWAYS_ALLOW_PATTERNS)
    if not args.ignore_data:
        allow_patterns.append("data/*")
    if not args.ignore_videos:
        allow_patterns.append("videos/*")

    local_dir = args.local_dir or os.path.join("/data", Path(args.repo_id).name)
    print(f"⬇️  Downloading {args.repo_id} -> {local_dir}")
    print(f"    allow_patterns: {allow_patterns}")

    # 1. Download the metadata, plus the parquet / video trees unless ignored
    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        token=hf_token,
    )

    # 2. Instantiate StreamingLeRobotDataset pointing to the local directory
    try:
        ds = StreamingLeRobotDataset(
            repo_id=args.repo_id,
            root=local_dir,
            streaming=True,
            buffer_size=args.buffer_size,
            shuffle=not args.no_shuffle,
            repo_type=args.repo_type,
            token=hf_token,
        )

        print("✅ Dataset loaded successfully!")
    except Exception as e:
        print("❌ Failed to load dataset:", str(e))
        return

    sample = next(iter(ds))
    print("Sample keys:", list(sample.keys()))
    print("Action shape:", sample["action"].shape)
    print("Language instruction:", sample.get("language_instruction", "N/A"))


if __name__ == "__main__":
    main()
