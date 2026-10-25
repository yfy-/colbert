#!/usr/bin/env python3

import argparse
import os
import sys
import glob
from tqdm import tqdm
from collections import defaultdict
import torch
from typing import Iterable, Tuple
from pathlib import Path

def iter_doc_tokens(index_dir: Path) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    i = 0
    while True:
        dlens_path = index_dir / f"dlens_{i}.pt"
        tokens_path = index_dir / f"tokens_{i}.pt"
        if not dlens_path.exists() and not tokens_path.exists():
            break

        if not dlens_path.exists():
            raise Exception(f"token path exists but not dlens: {dlens_path}")

        if not tokens_path.exists():
            raise Exception(f"dlens path exists but not tokens: {tokens_path}")

        batch_dlens = torch.load(dlens_path)
        batch_tokens = torch.load(tokens_path, map_location="cpu")
        assert sum(batch_dlens) == len(batch_tokens)
        offset = 0
        for dlen in batch_dlens:
            next_offset = offset + dlen
            yield batch_tokens[offset: next_offset]
            offset = next_offset

        i += 1

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
    for doc_tokens in tqdm(
            iterable=iter_doc_tokens(Path(args.index_dir)), desc="Processing document tokens"
    ):
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
