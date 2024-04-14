#!/usr/bin/env python3

import argparse
import torch
from collections import defaultdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trec_file")
    parser.add_argument("k", type=int)
    parser.add_argument("out_path")
    args = parser.parse_args()

    with open(args.trec_file) as tfs:
        trec_lines = tfs.readlines()

    result_dict = defaultdict(list)
    for tl in trec_lines:
        qid, _, cid, rank, _ = tl.rstrip().split(maxsplit=4)
        if int(rank) <= args.k:
            result_dict[int(qid)].append(int(cid))

    torch.save(
        {qid: torch.IntTensor(cids) for qid, cids in result_dict.items()},
        args.out_path,
    )
