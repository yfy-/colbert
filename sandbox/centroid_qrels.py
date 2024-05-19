#!/usr/bin/env python3

from collections import defaultdict
import json
from pathlib import Path

import torch

from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

if __name__ == "__main__":
    with Run().context(
        RunConfig(
            nranks=1,
            experiment="msmarco",
            name="triplets",
            overwrite=True,
        )
    ):

        searcher = Searcher(
            index="msmarco.nbits=2",
            config=ColBERTConfig(),
        )
        queries = Queries(
            path="your_path"
        )
        centroid_pids = torch.load(
            Path(searcher.index_config.root)
            / searcher.index_config.experiment
            / "indexes"
            / searcher.index_config.index_name
            / "your_centroid_pids_file"
        )

        triplets = searcher.search_centroid_triplet(queries, nprobe_bottom=0)
        qrels = defaultdict(dict)
        for qid, pos, _ in triplets:
            qrels[str(qid.item())][str(centroid_pids[pos.item()][0].item())] = 1

        with open("qrel.json", "w") as qfs:
            json.dump(qrels, qfs)
