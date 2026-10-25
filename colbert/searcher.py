import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from tqdm import tqdm

from colbert.data import Collection, Queries, Ranking, SparseQuery
from colbert.infra.config import ColBERTConfig
from colbert.infra.launcher import print_memory_stats
from colbert.infra.provenance import Provenance
from colbert.infra.run import Run
from colbert.modeling.checkpoint import Checkpoint
from colbert.search.index_storage import IndexScorer
from colbert.utils.utils import print_message

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
        # Sparcent search is opt-in even if the index was built with it.
        self.configure(sparcent=initial_config.sparcent)

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
            self.config,
            use_gpu=use_gpu,
        )

        # Sparcent queries are encoded with the SPLADE model used for the
        # centroid documents, loaded on first use.
        self.use_sparcent = (
            self.config.sparcent or self.config.sparcent_index is not None
        )
        self.splade = None
        self.splade_tokenizer = None
        self.splade_device = "cuda" if use_gpu else "cpu"

        print_memory_stats()

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def encode(
        self, text: TextQueries, full_length_search=False, ret_tokens=False
    ):
        queries = text if type(text) is list else [text]
        bsize = 128 if len(queries) > 128 else None

        self.checkpoint.query_tokenizer.query_maxlen = self.config.query_maxlen
        return self.checkpoint.queryFromText(
            queries,
            bsize=bsize,
            to_cpu=True,
            full_length_search=full_length_search,
            ret_tokens=ret_tokens,
        )

    def encode_sparse(self, queries: List[str]) -> List[SparseQuery]:
        """Encode queries into SPLADE term weights for the Sparcent index."""
        from splade.sparcent.queries import encode_sparse_queries
        from splade.sparcent.utils import load_tokenizer_and_model

        if self.splade is None:
            lora_dir = self.config.sparcent_splade_lora_dir
            self.splade_tokenizer, self.splade = load_tokenizer_and_model(
                self.config.sparcent_splade_model,
                Path(lora_dir) if lora_dir else None,
            )
            self.splade = self.splade.to(self.splade_device)
            self.splade.eval()

        return encode_sparse_queries(
            queries,
            self.splade,
            self.splade_tokenizer,
            device=self.splade_device,
        )

    def _sparse_queries(self, queries: Queries) -> Optional[Dict]:
        """SPLADE queries for Sparcent, unless given with the queries."""
        if not self.use_sparcent or queries.sparse_queries:
            return None

        print_message("Encoding queries with SPLADE..")
        sparse = self.encode_sparse(list(queries.values()))
        return dict(zip(queries.keys(), sparse))

    def search(self, text: str, k=10, filter_fn=None, full_length_search=False):
        Q, input_ids, attention_masks = self.encode(
            text, full_length_search=full_length_search, ret_tokens=True
        )
        sparse_query = self.encode_sparse([text])[0] if self.use_sparcent else None
        return self.dense_search(
            Q,
            input_ids[attention_masks == 1],
            k,
            filter_fn=filter_fn,
            sparse_query=sparse_query,
        )

    def search_all(
        self,
        queries: TextQueries,
        k=10,
        filter_fn=None,
        full_length_search=False,
    ):
        queries = Queries.cast(queries)
        queries_ = list(queries.values())

        Q, input_ids, attention_masks = self.encode(
            queries_, full_length_search=full_length_search, ret_tokens=True
        )

        return self._search_all_Q(
            queries,
            Q,
            input_ids,
            attention_masks,
            k,
            filter_fn=filter_fn,
            sparse_queries=self._sparse_queries(queries),
        )

    def search_all_timed(
        self,
        queries: TextQueries,
        k=10,
        filter_fn=None,
        full_length_search=False,
        num_run=2,
        sparse_then_dense=False,
        query_cache_path: Optional[Path] = None,
    ):
        queries = Queries.cast(queries)
        queries_ = list(queries.values())

        Q, input_ids, attention_masks = None, None, None
        if query_cache_path is not None and query_cache_path.exists():
            print_message(f"Loading cached Q from: {query_cache_path}")

            Q, input_ids, attention_masks = (
                torch.load(query_cache_path / "encoded.pt"),
                torch.load(query_cache_path / "input_ids.pt"),
                torch.load(query_cache_path / "attention_masks.pt"),
            )

        if Q is None:
            print_message("Encoding queries..")
            Q, input_ids, attention_masks = self.encode(
                queries_,
                full_length_search=full_length_search,
                ret_tokens=True,
            )

        if query_cache_path is not None and not query_cache_path.exists():
            print_message(f"Caching Q to cache file at: {query_cache_path}")
            query_cache_path.mkdir(parents=True)
            torch.save(Q, query_cache_path / "encoded.pt")
            torch.save(input_ids, query_cache_path / "input_ids.pt")
            torch.save(input_ids, query_cache_path / "attention_masks.pt")

        return self._search_all_Q_timed(
            queries,
            Q,
            input_ids,
            attention_masks,
            k,
            num_run,
            filter_fn=filter_fn,
            sparse_then_dense=sparse_then_dense,
            sparse_queries=self._sparse_queries(queries),
        )

    def _search_all_Q_timed(
        self,
        queries,
        Q,
        input_ids,
        attention_masks,
        k,
        num_run,
        filter_fn=None,
        sparse_then_dense=False,
        sparse_queries: Optional[Dict] = None,
    ):
        qids = list(queries.keys())
        num_q = Q.size(0)
        times = torch.zeros((num_q, num_run))
        all_scored_pids = []
        self.ranker.tot_time = 0
        for run in tqdm(range(num_run), desc="Run loop"):
            for query_idx in tqdm(range(num_q), desc="Query loop"):
                if sparse_queries is not None:
                    sparse_q = sparse_queries[qids[query_idx]]
                else:
                    sparse_q = queries.sparse_query(qids[query_idx])
                if sparse_then_dense:
                    if sparse_q is None:
                        raise ValueError(
                            "no sparse query for sparse then dense."
                        )

                    search_start = time.time()
                    pids, ranks, scores = self.sparse_then_dense_search(
                        Q[query_idx : query_idx + 1],
                        sparse_q,
                        k=k,
                    )
                else:
                    search_start = time.time()
                    pids, ranks, scores = self.dense_search(
                        Q[query_idx : query_idx + 1],
                        input_ids[query_idx : query_idx + 1][
                            attention_masks[query_idx : query_idx + 1] == 1
                        ],
                        k=k,
                        filter_fn=filter_fn,
                        sparse_query=sparse_q,
                    )

                times[query_idx][run] = (time.time() - search_start) * 1000
                if run == 0:
                    all_scored_pids.append(list(zip(pids, ranks, scores)))

        avg_cent_match = self.ranker.tot_time / num_run / num_q
        print_message(f"Avg cent elapsed ms: {avg_cent_match}")
        data = {qid: val for qid, val in zip(queries.keys(), all_scored_pids)}
        pivot_times = torch.min(times, dim=1).values
        provenance = Provenance()
        provenance.source = "Searcher::search_all_timed"
        provenance.queries = queries.provenance()
        provenance.config = self.config.export()
        provenance.k = k
        provenance.mean_centroid_latency = avg_cent_match
        provenance.mean_latency = torch.mean(pivot_times).item()
        provenance.median_latency = torch.median(pivot_times).item()
        tail_lat = torch.quantile(
            pivot_times, torch.tensor([0.95, 0.99]), interpolation="nearest"
        )
        provenance.p95_latency = tail_lat[0].item()
        provenance.p99_latency = tail_lat[1].item()
        print_message(
            f"#> Latency (ms, min of {num_run} runs) for k={k}: "
            f"mean={provenance.mean_latency:.1f} "
            f"p50={provenance.median_latency:.1f} "
            f"p95={provenance.p95_latency:.1f} "
            f"p99={provenance.p99_latency:.1f}"
        )
        return Ranking(data=data, provenance=provenance)

    def search_centroid_triplet(
        self, queries: TextQueries, nprobe_top=1, nprobe_bottom=1
    ):
        """Search for top and bottom centroids for query."""
        queries = Queries.cast(queries)
        queries_ = list(queries.values())

        Q = self.encode(queries_)
        qids = list(queries.keys())
        triplets = []
        for query_idx in tqdm(range(Q.size(0))):
            pairs = self.ranker.centroid_pair_rank(
                self.config,
                Q[query_idx : query_idx + 1],
                nprobe_top,
                nprobe_bottom,
            ).cpu()
            qid = torch.full((pairs.shape[0], 1), qids[query_idx])
            triplets.append(torch.cat((qid, pairs), dim=1))

        return torch.concat(triplets, dim=0)

    def _search_all_Q(
        self,
        queries,
        Q,
        input_ids,
        attention_masks,
        k,
        filter_fn=None,
        sparse_queries: Optional[Dict] = None,
    ):
        qids = list(queries.keys())
        search_start = time.time()
        all_scored_pids = [
            list(
                zip(
                    *self.dense_search(
                        Q[query_idx : query_idx + 1],
                        input_ids[query_idx : query_idx + 1][
                            attention_masks[query_idx : query_idx + 1] == 1
                        ],
                        k,
                        filter_fn=filter_fn,
                        sparse_query=(
                            sparse_queries[qids[query_idx]]
                            if sparse_queries is not None
                            else queries.sparse_query(qids[query_idx])
                        ),
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
        self,
        Q: torch.Tensor,
        input_tokens: torch.Tensor,
        k=10,
        filter_fn=None,
        sparse_query: SparseQuery = None,
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
            self.config, Q, input_tokens, filter_fn=filter_fn, sparse_query=sparse_query
        )

        return pids[:k], list(range(1, k + 1)), scores[:k]

    def sparse_then_dense_search(self, Q: torch.Tensor, sq: SparseQuery, k=10):
        pids = torch.from_numpy(
            self.ranker.sparcent_index.search_ndarray(
                sq,
                k=k,
                alpha=self.ranker.sparcent_alpha,
                beta=self.ranker.sparcent_beta,
            )
        ).int()
        pids, scores = self.ranker.rank_precomputed(self.config, Q, pids)
        return pids[:k], list(range(1, k + 1)), scores[:k]
