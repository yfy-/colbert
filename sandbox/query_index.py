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
                save_centroids=False,
                precomputed_centroid_path="splade_distil_dev_small_unq_3.pt",
                precomputed_num_cand=10
            ),
        )
        queries = Queries(
            path="/home/clownland/Datasets/msmarco/ir-datasets/"
            "queries.dev.small.tsv"
        )
        ranking = searcher.search_all_timed(queries, k=10, num_run=3)
        ranking.save("msmarco_dev_small.nbits=2.ranking.tsv")
