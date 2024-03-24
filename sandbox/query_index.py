#!/usr/bin/env python3

from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

if __name__ == "__main__":
    with Run().context(
        RunConfig(
            nranks=1,
            experiment="msmarco",
            name="splade_centroids",
            overwrite=True,
            # gpus=0,
        )
    ):

        searcher = Searcher(
            index="msmarco.nbits=2",
            config=ColBERTConfig(
                save_centroids=True,
                precomputed_centroid_path="splade_distil_dev_small_unq_3.pt",
            ),
        )
        queries = Queries(
            path="/home/clownland/Datasets/msmarco/ir-datasets/"
            "queries.dev.small.tsv"
        )
        ranking = searcher.search_all(queries, k=10)
        ranking.save("msmarco_dev_small.nbits=2.ranking.tsv")
