#!/usr/bin/env python3

import argparse
import sys
from collections import defaultdict

import torch


def main(output_path: str):
    doc2queries = defaultdict(list)
    for line in sys.stdin:
        qid, did, _, _ = line.rstrip.split("\t")
        doc2queries[did].append(qid)

    torch.save(doc2queries, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path")
    args = parser.parse_args()
    exit(main(args.output_path))
