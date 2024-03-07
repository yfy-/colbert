#!/usr/bin/env python3

from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="msmarco", name="splade_centroids")):

        config = ColBERTConfig()
        searcher = Searcher(index="msmarco.nbits=2", config=config)
        queries = Queries("/home/clownland/Datasets/msmarco/queries_5k.dev.tsv")
        ranking = searcher.search_all(queries, k=1000)
        ranking.save("msmarco_dev_5k.nbits=2.ranking.tsv")
