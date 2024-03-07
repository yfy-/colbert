#!/usr/bin/env python3

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="msmarco")):

        config = ColBERTConfig(nbits=2)
        indexer = Indexer(
            checkpoint="/home/clownland/Projects/ColBERT/colbertv2.0",
            config=config
        )
        indexer.index(
            name="msmarco.nbits=2",
            collection="/home/clownland/Datasets/msmarco_100k.tsv",
            overwrite=True
        )
