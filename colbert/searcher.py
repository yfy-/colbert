import os
import torch
import json

from tqdm import tqdm
from typing import Union, Optional

from colbert.data import Collection, Queries, Ranking

from colbert.modeling.checkpoint import Checkpoint
from colbert.search.index_storage import IndexScorer

from colbert.infra.provenance import Provenance
from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert.infra.launcher import print_memory_stats
from colbert.utils.utils import print_message

import time

TextQueries = Union[str, "list[str]", "dict[int, str]", Queries]


class Searcher:
    def __init__(self, index, checkpoint=None, collection=None, config=None):
        print_memory_stats()

        initial_config = ColBERTConfig.from_existing(config, Run().config)
        default_index_root = initial_config.index_root_
        self.index = os.path.join(default_index_root, index)
        self.index_config = ColBERTConfig.load_from_index(self.index)

        self.checkpoint = checkpoint or self.index_config.checkpoint
        self.checkpoint_config = ColBERTConfig.load_from_checkpoint(
            self.checkpoint
        )
        self.config = ColBERTConfig.from_existing(
            self.checkpoint_config, self.index_config, initial_config
        )

        self.collection = Collection.cast(collection or self.config.collection)
        self.configure(checkpoint=self.checkpoint, collection=self.collection)

        self.checkpoint = Checkpoint(
            self.checkpoint, colbert_config=self.config
        )
        use_gpu = self.config.gpus > 0
        print_message(f"GPU status: {use_gpu}")
        if use_gpu:
            self.checkpoint = self.checkpoint.cuda()

        self.ranker = IndexScorer(
            self.index,
            use_gpu=use_gpu,
            precomputed_centroid_path=self.config.precomputed_centroid_path,
            precomputed_num_cand=self.config.precomputed_num_cand,
            filter_pids_maxscore=self.config.filter_pids_maxscore,
            maxscore_quantile=self.config.maxscore_quantile,
        )

        print_memory_stats()

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def encode(self, text: TextQueries, full_length_search=False):
        queries = text if type(text) is list else [text]
        bsize = 128 if len(queries) > 128 else None

        self.checkpoint.query_tokenizer.query_maxlen = self.config.query_maxlen
        Q = self.checkpoint.queryFromText(
            queries,
            bsize=bsize,
            to_cpu=True,
            full_length_search=full_length_search,
        )

        return Q

    def search(self, text: str, k=10, filter_fn=None, full_length_search=False):
        Q = self.encode(text, full_length_search=full_length_search)
        return self.dense_search(Q, k, filter_fn=filter_fn)

    def search_all(
        self,
        queries: TextQueries,
        k=10,
        filter_fn=None,
        full_length_search=False,
    ):
        queries = Queries.cast(queries)
        queries_ = list(queries.values())

        Q = self.encode(queries_, full_length_search=full_length_search)

        return self._search_all_Q(queries, Q, k, filter_fn=filter_fn)

    def search_all_timed(
        self,
        queries: TextQueries,
        k=10,
        filter_fn=None,
        full_length_search=False,
        num_run=2,
    ):
        queries = Queries.cast(queries)
        queries_ = list(queries.values())

        Q = self.encode(queries_, full_length_search=full_length_search)

        return self._search_all_Q_timed(
            queries, Q, k, num_run, filter_fn=filter_fn
        )

    def _search_all_Q_timed(self, queries, Q, k, num_run, filter_fn=None):
        qids = list(queries.keys())
        num_q = Q.size(0)
        times = torch.zeros((num_q, num_run))
        all_scored_pids = []
        for run in tqdm(range(num_run), desc="Run loop"):
            for query_idx in tqdm(range(num_q), desc="Query loop"):
                search_start = time.time()
                pids, ranks, scores = self.dense_search(
                    Q[query_idx : query_idx + 1],
                    k,
                    filter_fn=filter_fn,
                    qid=qids[query_idx],
                )
                times[query_idx][run] = (time.time() - search_start) * 1000
                if run == 0:
                    all_scored_pids.append(list(zip(pids, ranks, scores)))

        data = {qid: val for qid, val in zip(queries.keys(), all_scored_pids)}
        pivot_times = torch.min(times, dim=1).values
        provenance = Provenance()
        provenance.source = "Searcher::search_all_timed"
        provenance.queries = queries.provenance()
        provenance.config = self.config.export()
        provenance.k = k
        provenance.mean_latency = torch.mean(pivot_times).item()
        provenance.median_latency = torch.median(pivot_times).item()
        tail_lat = torch.quantile(
            pivot_times, torch.tensor([0.95, 0.99]), interpolation="nearest"
        )
        provenance.p95_latency = tail_lat[0].item()
        provenance.p99_latency = tail_lat[1].item()
        return Ranking(data=data, provenance=provenance)

    def search_centroid_triplet(
        self, queries: TextQueries, nprobe_top=1, nprobe_bottom=1
    ):
        """Search for top and bottom centroids for query."""
        queries = Queries.cast(queries)
        queries_ = list(queries.values())

        Q = self.encode(queries_)
        qids = list(queries.keys())
        triplets = None
        for query_idx in tqdm(range(Q.size(0))):
            pairs = self.ranker.centroid_pair_rank(
                self.config,
                Q[query_idx : query_idx + 1],
                nprobe_top,
                nprobe_bottom,
            ).cpu()
            qid = torch.full((pairs.shape[0], 1), qids[query_idx])
            triplet = torch.cat((qid, pairs), dim=1)
            if triplets is None:
                triplets = triplet
            else:
                triplets = torch.cat((triplets, triplet), dim=0)

        return triplets

    def _search_all_Q(self, queries, Q, k, filter_fn=None):
        qids = list(queries.keys())
        search_start = time.time()
        all_scored_pids = [
            list(
                zip(
                    *self.dense_search(
                        Q[query_idx : query_idx + 1],
                        k,
                        filter_fn=filter_fn,
                        qid=qids[query_idx],
                    )
                )

            )
            for query_idx in tqdm(range(Q.size(0)))
        ]
        elapsed = time.time() - search_start
        n = Q.size(0)
        print_message(f"Avg retrieval ms: {elapsed * 1000 / n}")
        avg_cent_match = self.ranker.tot_time / n
        avg_cent_size = self.ranker.tot_cells / n
        print_message(f"Avg cent elapsed ms: {avg_cent_match}")
        print_message(f"Avg cent retrieved: {avg_cent_size}")
        if self.config.save_centroids:
            centroids_per_query = {}
            for i, qid in enumerate(queries.keys()):
                centroids_per_query[qid] = self.ranker.cells_retrieved[i]

            with Run().open("retrieved_centroids.json", "w") as crfs:
                json.dump(centroids_per_query, crfs)

        data = {qid: val for qid, val in zip(queries.keys(), all_scored_pids)}

        provenance = Provenance()
        provenance.source = "Searcher::search_all"
        provenance.queries = queries.provenance()
        provenance.config = self.config.export()
        provenance.k = k

        return Ranking(data=data, provenance=provenance)

    def dense_search(
        self, Q: torch.Tensor, k=10, filter_fn=None, qid: Optional[int] = None
    ):
        if k <= 10:
            if self.config.ncells is None:
                self.configure(ncells=1)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.5)
            if self.config.ndocs is None:
                self.configure(ndocs=256)
        elif k <= 100:
            if self.config.ncells is None:
                self.configure(ncells=2)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.45)
            if self.config.ndocs is None:
                self.configure(ndocs=1024)
        else:
            if self.config.ncells is None:
                self.configure(ncells=4)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.4)
            if self.config.ndocs is None:
                self.configure(ndocs=max(k * 4, 4096))

        pids, scores = self.ranker.rank(
            self.config, Q, filter_fn=filter_fn, qid=qid
        )

        return pids[:k], list(range(1, k + 1)), scores[:k]
