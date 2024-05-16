#!/usr/bin/env python3

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
        collection = dict()
        with open("your_msmarco_collection") as cfs:
            collection_lines = cfs.readlines()
            for cl in collection_lines:
                cl = cl.rstrip()
                docid, doc = cl.split("\t")
                collection[int(docid)] = doc

        triplets = searcher.search_centroid_triplet(queries)
        with open("your_output_triplets", "w") as tfs:
            for qid, pos, neg in triplets:
                query = queries[qid.item()]
                pos_psg = collection[centroid_pids[pos.item()][0].item()]
                neg_psg = collection[centroid_pids[neg.item()][0].item()]
                tfs.write(f"{query}\t{pos_psg}\t{neg_psg}\n")
