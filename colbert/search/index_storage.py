import torch
from typing import Optional

from colbert.utils.utils import flatten, print_message

from colbert.indexing.loaders import load_doclens
from colbert.indexing.codecs.residual_embeddings_strided import (
    ResidualEmbeddingsStrided,
)

from colbert.search.strided_tensor import StridedTensor
from colbert.search.candidate_generation import CandidateGeneration

from .index_loader import IndexLoader
from colbert.modeling.colbert import (
    colbert_score,
    colbert_score_packed,
    colbert_score_reduce,
)

from math import ceil

import os
import pathlib
from torch.utils.cpp_extension import load


class IndexScorer(IndexLoader, CandidateGeneration):
    filter_pids_file = "filter_pids.cpp"

    def __init__(
        self,
        index_path,
        use_gpu=True,
        precomputed_centroid_path: str = None,
        precomputed_num_cand: int = None,
        filter_pids_maxscore: bool = False,
        maxscore_quantile: float = None
    ):
        super().__init__(index_path=index_path, use_gpu=use_gpu)
        self.tot_time = 0
        self.tot_cells = 0
        self.cells_retrieved = []

        # search params
        self.precomputed_centroids = None
        self.get_cells_func = self.get_cells
        self.filter_pids_maxscore = filter_pids_maxscore
        self.maxscore_quantile = maxscore_quantile
        if filter_pids_maxscore:
            self.get_cells_func = self.get_cells_term_ub_scores
            self.__class__.filter_pids_file = "filter_pids_maxscore.cpp"
        elif precomputed_centroid_path is not None:
            if precomputed_num_cand is None or precomputed_num_cand < 1:
                raise ValueError("invalid 'precomputed_num_cand'")
            self.precomputed_centroids = torch.load(precomputed_centroid_path)
            self.precomputed_num_cand = precomputed_num_cand
            self.get_cells_func = self.get_cells_precomputed

        IndexScorer.try_load_torch_extensions(use_gpu)

        self.embeddings_strided = ResidualEmbeddingsStrided(
            self.codec, self.embeddings, self.doclens
        )

    @classmethod
    def try_load_torch_extensions(cls, use_gpu):
        if hasattr(cls, "loaded_extensions") or use_gpu:
            return

        print_message(
            f"Loading filter_pids_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        filter_pids_cpp = load(
            name="filter_pids_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(),
                    cls.filter_pids_file,
                ),
            ],
            extra_cflags=["-O3"],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False")
            == "True",
        )
        cls.filter_pids = filter_pids_cpp.filter_pids_cpp

        print_message(
            f"Loading decompress_residuals_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        decompress_residuals_cpp = load(
            name="decompress_residuals_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(),
                    "decompress_residuals.cpp",
                ),
            ],
            extra_cflags=["-O3"],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False")
            == "True",
        )
        cls.decompress_residuals = (
            decompress_residuals_cpp.decompress_residuals_cpp
        )

        cls.loaded_extensions = True

    def lookup_eids(self, embedding_ids, codes=None, out_device="cuda"):
        return self.embeddings_strided.lookup_eids(
            embedding_ids, codes=codes, out_device=out_device
        )

    def lookup_pids(self, passage_ids, out_device="cuda", return_mask=False):
        return self.embeddings_strided.lookup_pids(passage_ids, out_device)

    def retrieve(self, config, Q, qid: Optional[int] = None):
        Q = Q[
            :, : config.query_maxlen
        ]  # NOTE: Candidate generation uses only the query tokens
        embedding_ids, centroid_scores, term_ub_scores = (
            self.generate_candidates(config, Q, qid=qid)
        )

        return embedding_ids, centroid_scores, term_ub_scores

    def embedding_ids_to_pids(self, embedding_ids):
        all_pids = torch.unique(
            self.emb2pid[embedding_ids.long()].cuda(), sorted=False
        )
        return all_pids

    def rank(self, config, Q, filter_fn=None, qid: Optional[int] = None):
        with torch.inference_mode():
            pids, centroid_scores, term_ub_scores = self.retrieve(
                config, Q, qid=qid
            )

            if filter_fn is not None:
                pids = filter_fn(pids)
                if len(pids) == 0:
                    return [], []

            scores, pids = self.score_pids(
                config, Q, pids, centroid_scores, term_ub_scores
            )

            scores_sorter = scores.sort(descending=True)
            pids, scores = (
                pids[scores_sorter.indices].tolist(),
                scores_sorter.values.tolist(),
            )

            return pids, scores

    def score_pids(self, config, Q, pids, centroid_scores, term_ub_scores):
        """
        Always supply a flat list or tensor for `pids`.

        Supply sizes Q = (1 | num_docs, *, dim) and D = (num_docs, *, dim).
        If Q.size(0) is 1, the matrix will be compared with all passages.
        Otherwise, each query matrix will be compared against the *aligned* passage.
        """

        # TODO: Remove batching?
        batch_size = 2**20

        if self.use_gpu:
            centroid_scores = centroid_scores.cuda()

        if self.use_gpu or not self.filter_pids_maxscore:
            idx = (
                centroid_scores.max(-1).values
                >= config.centroid_score_threshold
            )
        else:
            idx = None

        if self.use_gpu:
            approx_scores = []

            # Filter docs using pruned centroid scores
            for i in range(0, ceil(len(pids) / batch_size)):
                pids_ = pids[i * batch_size : (i + 1) * batch_size]
                codes_packed, codes_lengths = (
                    self.embeddings_strided.lookup_codes(pids_)
                )
                idx_ = idx[codes_packed.long()]
                pruned_codes_strided = StridedTensor(
                    idx_, codes_lengths, use_gpu=self.use_gpu
                )
                pruned_codes_padded, pruned_codes_mask = (
                    pruned_codes_strided.as_padded_tensor()
                )
                pruned_codes_lengths = (
                    pruned_codes_padded * pruned_codes_mask
                ).sum(dim=1)
                codes_packed_ = codes_packed[idx_]
                approx_scores_ = centroid_scores[codes_packed_.long()]
                if approx_scores_.shape[0] == 0:
                    approx_scores.append(
                        torch.zeros(
                            (len(pids_),), dtype=approx_scores_.dtype
                        ).cuda()
                    )
                    continue
                approx_scores_strided = StridedTensor(
                    approx_scores_, pruned_codes_lengths, use_gpu=self.use_gpu
                )
                approx_scores_padded, approx_scores_mask = (
                    approx_scores_strided.as_padded_tensor()
                )
                approx_scores_ = colbert_score_reduce(
                    approx_scores_padded, approx_scores_mask, config
                )
                approx_scores.append(approx_scores_)
            approx_scores = torch.cat(approx_scores, dim=0)
            assert approx_scores.is_cuda, approx_scores.device
            if config.ndocs < len(approx_scores):
                pids = pids[torch.topk(approx_scores, k=config.ndocs).indices]

            # Filter docs using full centroid scores
            codes_packed, codes_lengths = self.embeddings_strided.lookup_codes(
                pids
            )
            approx_scores = centroid_scores[codes_packed.long()]
            approx_scores_strided = StridedTensor(
                approx_scores, codes_lengths, use_gpu=self.use_gpu
            )
            approx_scores_padded, approx_scores_mask = (
                approx_scores_strided.as_padded_tensor()
            )
            approx_scores = colbert_score_reduce(
                approx_scores_padded, approx_scores_mask, config
            )
            if config.ndocs // 4 < len(approx_scores):
                pids = pids[
                    torch.topk(approx_scores, k=(config.ndocs // 4)).indices
                ]
        else:
            filter_pids_args = [
                pids,
                centroid_scores,
                self.embeddings.codes,
                self.doclens,
                self.embeddings_strided.codes_strided.offsets,
                idx,
                config.ndocs,
            ]
            if self.filter_pids_maxscore:
                # Remove idx parameter
                del filter_pids_args[5]
                filter_pids_args.append(term_ub_scores)

            pids = IndexScorer.filter_pids(*filter_pids_args)

        # Rank final list of docs using full approximate embeddings (including residuals)
        if self.use_gpu:
            D_packed, D_mask = self.lookup_pids(pids)
        else:
            D_packed = IndexScorer.decompress_residuals(
                pids,
                self.doclens,
                self.embeddings_strided.codes_strided.offsets,
                self.codec.bucket_weights,
                self.codec.reversed_bit_map,
                self.codec.decompression_lookup_table,
                self.embeddings.residuals,
                self.embeddings.codes,
                self.codec.centroids,
                self.codec.dim,
                self.codec.nbits,
            )
            D_packed = torch.nn.functional.normalize(
                D_packed.to(torch.float32), p=2, dim=-1
            )
            D_mask = self.doclens[pids.long()]

        if Q.size(0) == 1:
            return colbert_score_packed(Q, D_packed, D_mask, config), pids

        D_strided = StridedTensor(D_packed, D_mask, use_gpu=self.use_gpu)
        D_padded, D_lengths = D_strided.as_padded_tensor()

        return colbert_score(Q, D_padded, D_lengths, config), pids
