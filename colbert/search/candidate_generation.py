import time
from typing import Optional

import torch

from colbert.data import SparseQuery
from colbert.search.strided_tensor import StridedTensor


class CandidateGeneration:

    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu

    def get_cells(self, Q, ncells, sparse_query: Optional[SparseQuery] = None):
        _ = sparse_query
        scores = self.codec.centroids @ Q.T
        if ncells == 1:
            cells = scores.argmax(dim=0, keepdim=True).permute(1, 0)
        else:
            cells = scores.topk(ncells, dim=0, sorted=False).indices.permute(
                1, 0
            )  # (32, ncells)
        cells = cells.flatten().contiguous()  # (32 * ncells,)
        cells = cells.unique(sorted=False)
        return cells, scores, None

    def get_cells_sparcent(
        self, Q, ncells, sparse_query: Optional[SparseQuery] = None
    ):
        if sparse_query is None:
            raise ValueError("sparse query cannot be null.")

        cells = self.sparcent_index.search_quantized(
            sparse_query,
            k=self.sparcent_k_score,
            alpha=self.sparcent_alpha,
            beta=self.sparcent_beta,
        )[0]
        cells = torch.LongTensor(list(map(int, cells)))
        if self.use_gpu:
            cells = cells.to("cuda")

        selected_scores = self.codec.centroids[cells] @ Q.T
        self.pre_alloc_scores.zero_()
        self.pre_alloc_scores[cells] = selected_scores
        rerank_scores = selected_scores[: self.sparcent_k_rerank]
        if ncells == 1:
            reranked_idxs = rerank_scores.argmax(dim=0, keepdim=True).permute(
                1, 0
            )
        else:
            reranked_idxs = rerank_scores.topk(
                ncells, dim=0, sorted=False
            ).indices.permute(1, 0)

        reranked_idxs = reranked_idxs.flatten().unique(sorted=False)
        return cells[reranked_idxs], self.pre_alloc_scores, None

    def get_cells_term_ub_scores(
        self, Q, ncells, sparse_query: Optional[SparseQuery] = None
    ):
        _ = sparse_query
        scores = self.codec.centroids @ Q.T
        if ncells == 1:
            scores_topk = scores.max(dim=0, keepdim=True)
        else:
            scores_topk = scores.topk(ncells, dim=0)

        cells = scores_topk.indices.flatten().contiguous().unique(sorted=False)

        # Sort cells by their importance
        cell_importance = torch.sum(scores[cells], dim=1)
        cells = cells[cell_importance.argsort(descending=True)]
        # ub_scores =  scores.quantile(
        #     self.maxscore_quantile, dim=0, interpolation="nearest"
        # )

        ub_scores = scores_topk.values[0, :]
        ub_scores_sorted = ub_scores.sort(descending=True)
        scores = scores[:, ub_scores_sorted.indices]
        ub_scores = ub_scores_sorted.values.contiguous()

        return cells, scores, ub_scores

    def generate_candidate_eids(self, Q, ncells):
        cells, scores = self.get_cells_func(Q, ncells)

        eids, cell_lengths = self.ivf.lookup(
            cells
        )  # eids = (packedlen,)  lengths = (32 * ncells,)
        eids = eids.long()
        if self.use_gpu:
            eids = eids.cuda()
        return eids, scores

    def generate_candidate_pids(
        self, Q, ncells, sparse_query: SparseQuery = None
    ):
        cand_beg = time.time()
        cells, scores, term_ub_scores = self.get_cells_func(
            Q, ncells, sparse_query=sparse_query
        )
        # cells, scores = self.get_cells(Q, ncells)
        elapsed_ms = (time.time() - cand_beg) * 1000
        self.tot_time += elapsed_ms
        self.tot_cells += len(cells)
        # self.cells_retrieved.append(cells.tolist())

        pids, cell_lengths = self.ivf.lookup(cells)
        if self.use_gpu:
            pids = pids.cuda()
        return pids, scores, term_ub_scores

    def generate_candidate_scores(self, Q, eids):
        E = self.lookup_eids(eids)
        if self.use_gpu:
            E = E.cuda()
        return (Q.unsqueeze(0) @ E.unsqueeze(2)).squeeze(-1).T

    def generate_candidates(
        self, config, Q, sparse_query: Optional[SparseQuery] = None
    ):
        ncells = config.ncells

        assert isinstance(self.ivf, StridedTensor)

        Q = Q.squeeze(0)
        if self.use_gpu:
            Q = Q.cuda().half()
        assert Q.dim() == 2

        pids, centroid_scores, term_ub_scores = self.generate_candidate_pids(
            Q, ncells, sparse_query=sparse_query
        )

        if not self.filter_pids_maxscore:
            sorter = pids.sort()
            pids = sorter.values

            pids, pids_counts = torch.unique_consecutive(
                pids, return_counts=True
            )
            if self.use_gpu:
                pids, pids_counts = pids.cuda(), pids_counts.cuda()

        return pids, centroid_scores, term_ub_scores

    def generate_candidates_pairs(
        self, Q, nprobe_top, nprobe_bottom
    ) -> torch.IntTensor:
        assert nprobe_top > 0

        Q = Q.squeeze(0)
        if self.use_gpu:
            Q = Q.cuda().half()
        assert Q.dim() == 2

        scores = self.codec.centroids @ Q.T
        if nprobe_top == 1:
            top_cells = scores.argmax(dim=0, keepdim=True)
        else:
            top_cells = scores.topk(nprobe_top, dim=0, sorted=False).indices

        if nprobe_bottom <= 0:
            bottom_cells = None
        elif nprobe_bottom == 1:
            bottom_cells = scores.argmin(dim=0, keepdim=True)
        else:
            bottom_cells = scores.topk(
                nprobe_bottom, dim=0, sorted=False, largest=False
            ).indices

        top_cells = top_cells.flatten().unique(sorted=False)
        if bottom_cells is None:
            top_cells = top_cells.unsqueeze(dim=1)
            return torch.cat(
                (top_cells, torch.full_like(top_cells, -1, dtype=torch.int32)),
                dim=1,
            )

        bottom_cells = bottom_cells.flatten().unique(sorted=False)
        top_bottom_scores = (
            self.codec.centroids[top_cells]
            @ self.codec.centroids[bottom_cells].T
        )
        bottom_cells = bottom_cells[top_bottom_scores.argmin(dim=1)]
        return torch.cat(
            (top_cells.unsqueeze(dim=1), bottom_cells.unsqueeze(dim=1)), dim=1
        )
