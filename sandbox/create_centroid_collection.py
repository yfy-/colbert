"""Create a collection for centroids."""

import argparse
import logging
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
    args = parser.parse_args()

    # We won't do anything useful with GPU.
    sample_pids = torch.load(args.sample_pids).to("cpu")

    unq_pids = torch.unique(sample_pids)
    sp_cursor = 0

    # First element can be -1 for not found documents
    if unq_pids[sp_cursor] == -1:
        sp_cursor += 1

    docid2doc = dict()
    line_idx = 0

    with tqdm(desc="Collect required documents") as pbar:
        for line in args.in_collection:
            docid, doc = line.rstrip().split("\t")
            try:
                # Unfortunately have to do this, since the compressed
                # collection file has an unexpected prefix before the
                # docid which causes docid to be wrong. In case we cannot
                # cast that to int we will use the line_idx. This is ok
                # because docid's already start from 0 and increment by 1.
                docid = int(docid)
            except ValueError:
                logging.warning(
                    f"{docid} not an int, will use line idx as docid"
                )
                docid = line_idx

            line_idx += 1
            pbar.update(1)
            if docid != unq_pids[sp_cursor].item():
                continue

            docid2doc[docid] = doc
            sp_cursor += 1

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
