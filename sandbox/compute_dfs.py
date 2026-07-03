import argparse
import os
import sys
import glob
from tqdm import tqdm
from collections import defaultdict
import torch

def main():
    parser = argparse.ArgumentParser(
        description="Compute IDF's of a ColBERT index."
    )
    parser.add_argument(
        "index_dir",
        help="Path to index directory",
    )
    parser.add_argument(
        "output_path",
        help="Output tensor path",
    )
    args = parser.parse_args()
    dfs = defaultdict(int)
    max_token_id = 0
    for tokens_path in tqdm(
            iterable=glob.glob(f"{args.index_dir}/tokens_*.pt"), desc="Reading token files"
    ):
        tokens = torch.load(tokens_path, map_location="cpu")
        for doc_tokens in tokens:
            tids = doc_tokens.tolist()
            for tid in set(tids):
                max_token_id = max(max_token_id, tid)
                dfs[tid] += 1


    out_tensor = torch.zeros(max_token_id + 1, dtype=torch.long)
    for token_id, df in dfs.items():
        out_tensor[token_id] = df

    torch.save(out_tensor, args.output_path)

if __name__ == "__main__":
    main()
