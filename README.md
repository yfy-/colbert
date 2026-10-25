# ColBERT (fork)

A research fork of ColBERTv2/PLAID with the code of the works below.
The options they add are in `colbert/infra/config/settings.py`.

## Sparcent

[Sparcent: faster multi-vector dense retrieval with sparse centroid
indices](https://doi.org/10.1007/s10115-026-02907-x) (Knowledge and
Information Systems, 2026). Setup, command line tools and the paper's
experiments are in the [sparcent](https://github.com/yfy-/sparcent)
repository.

- **Index building** (`colbert/sparcent/`). With
  `ColBERTConfig(sparcent=True)` the `Indexer` builds, after the
  ColBERT index, a sparse centroid index in `<index>/sparcent/`: the
  passages of each centroid are ordered (`sparcent_sample_method`:
  `count`, `similar` or `mean_similar`), the SPLADE vectors of the
  first `sparcent_nearest_percent` of them are averaged and pruned to
  an L1 submass (`sparcent_pruning_submass`), then indexed by PISA,
  reordered by recursive graph bisection (`sparcent_bp_depth`) and
  converted to a BMP index (`centroids.bmp`).
- **Search.** `Searcher(..., config=ColBERTConfig(sparcent=True,
  ...))` retrieves `sparcent_k_score` centroids with BMP
  (`sparcent_alpha`, `sparcent_beta`) and re-ranks the first
  `sparcent_k_rerank` with dense scores instead of scoring all
  centroids. Queries are encoded with both ColBERT and SPLADE (the
  model used at indexing).

## Static pruning

Static pruning of document token embeddings at indexing time
(`static_pruning_method`, `static_pruning_kwargs`; the methods are in
`colbert/indexing/collection_encoder.py`), including methods that use
a query log.

## Other additions

- **Latency measurement.** `Searcher.search_all_timed(queries, k,
  num_run)` runs each query `num_run` times and reports
  mean/p50/p95/p99 of the per-query minimum latency (log and ranking
  `.meta`). The `gpus` setting is respected, e.g. `RunConfig(gpus=0)`
  for CPU search.
- **Experiment options:** sparse-then-dense retrieval
  (`sparse_then_dense=True` with a full-collection SPLADE BMP
  `sparcent_index`), IDF query token pruning (`token_dfs_path`,
  `token_topk`), MaxScore filtering (`filter_pids_maxscore`),
  embedding and query caches (`encode_cache_path`,
  `query_cache_path`).
- `utility/evaluate/msmarco_passages.py` also reports NDCG@10 and
  Recall@10/100 and reads TREC runs;
  `utility/evaluate/lotte_evaluate.py` evaluates LoTTE. `scripts/`
  holds one-off experiment scripts (not maintained).

## Installation

```bash
pip install -e ".[sparcent]"   # without [sparcent] for plain ColBERT/PLAID
```

The `sparcent` extra installs the
[splade](https://github.com/yfy-/splade) and
[BMP](https://github.com/yfy-/BMP) forks (BMP's Python bindings are
built with Rust). Building a Sparcent index also needs the PISA tools
(`parse_collection`, `invert`, `reorder-docids`, `lexicon`) and ciff's
`pisa2ciff` on `PATH` (or in `sparcent_pisa_bin_dir`). The sparcent
repository's `setup.sh` sets all of them up with the versions used for
the paper.

## Sparcent usage

```python
from colbert import Indexer, Searcher
from colbert.infra import ColBERTConfig, Run, RunConfig

with Run().context(RunConfig(nranks=1, experiment="msmarco")):
    config = ColBERTConfig(nbits=2, sparcent=True, sparcent_nearest_percent=0.01,
                           sparcent_pruning_submass=0.05, sparcent_bp_depth=13)
    Indexer(checkpoint="colbert-ir/colbertv2.0", config=config).index(
        name="msmarco.nbits=2", collection="collection.tsv")

with Run().context(RunConfig(nranks=1, experiment="msmarco", gpus=0)):
    searcher = Searcher(index="msmarco.nbits=2", config=ColBERTConfig(
        sparcent=True, sparcent_k_score=5000, sparcent_k_rerank=3000,
        sparcent_alpha=0.85, sparcent_beta=0.9))
    ranking = searcher.search_all_timed("queries.dev.small.tsv", k=10, num_run=3)
```

## Acknowledgements

This is a fork of
[ColBERT](https://github.com/stanford-futuredata/ColBERT) by the
Stanford FutureData group; all credit for ColBERT, ColBERTv2 and PLAID
goes to its authors. See the original repository for documentation,
and cite [ColBERT](https://arxiv.org/abs/2004.12832) (SIGIR'20),
[ColBERTv2](https://arxiv.org/abs/2112.01488) (NAACL'22) and
[PLAID](https://arxiv.org/abs/2205.09707) (CIKM'22).
