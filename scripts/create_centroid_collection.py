#!/usr/bin/env python3

"""Create a collection for centroids."""

import argparse
import sys

import torch
from tqdm import tqdm


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_pids")
    parser.add_argument(
        "in_collection", nargs="?", type=argparse.FileType(), default=sys.stdin
    )
    parser.add_argument(
        "out_collection",
        nargs="?",
        type=argparse.FileType("w"),
        default=sys.stdout,
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=0,
        help="Use only top-k passages for each centroid. 0 is use all.",
    )
    args = parser.parse_args()

    # We won't do anything useful with GPU.
    sample_pids = torch.load(args.sample_pids).to("cpu")
    if args.top_k > 0:
        sample_pids = sample_pids[:, :args.top_k]

    unq_pids = torch.unique(sample_pids)
    sp_cursor = 0

    # First element can be -1 for not found documents
    if unq_pids[sp_cursor].item() == -1:
        sp_cursor += 1

    unq_len = len(unq_pids)
    docid2doc = dict()
    with tqdm(desc="Collect required documents") as pbar:
        for line in args.in_collection:
            docid, doc = line.rstrip().split("\t")
            docid = int(docid)
            pbar.update(1)

            if sp_cursor == unq_len:
                break

            if docid != unq_pids[sp_cursor].item():
                continue

            docid2doc[docid] = doc
            sp_cursor += 1

    args.in_collection.close()
    with tqdm(total=sample_pids.shape[0], desc="Create centroid docs") as pbar:
        for cent_id, docids in enumerate(sample_pids):
            concat_doc = ""
            for did in docids:
                did = did.item()
                if did == -1:
                    continue

                concat_doc += docid2doc[did]

            args.out_collection.write(f"{cent_id}\t{concat_doc}\n")
            pbar.update(1)


if __name__ == "__main__":
    exit(_main())
