import torch
import time
from typing import Optional

from colbert.search.strided_tensor import StridedTensor
from .strided_tensor_core import _create_mask, _create_view


class CandidateGeneration:

    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu

    def get_cells(self, Q, ncells, qid: Optional[int] = None):
        scores = (self.codec.centroids @ Q.T)
        if ncells == 1:
            cells = scores.argmax(dim=0, keepdim=True).permute(1, 0)
        else:
            cells = scores.topk(ncells, dim=0, sorted=False).indices.permute(1, 0)  # (32, ncells)
        cells = cells.flatten().contiguous()  # (32 * ncells,)
        cells = cells.unique(sorted=False)
        return cells, scores

    def get_cells_precomputed(self, Q, ncells, qid: int):
        cells = self.precomputed_centroids[qid].contiguous()
        selected_scores = self.codec.centroids[cells] @ Q.T
        scores = torch.zeros((self.codec.centroids.shape[0], Q.shape[0]))
        scores[cells] = selected_scores
        return cells[:self.precomputed_num_cand], scores

    def generate_candidate_eids(self, Q, ncells):
        cells, scores = self.get_cells_func(Q, ncells)

        eids, cell_lengths = self.ivf.lookup(cells)  # eids = (packedlen,)  lengths = (32 * ncells,)
        eids = eids.long()
        if self.use_gpu:
            eids = eids.cuda()
        return eids, scores

    def generate_candidate_pids(self, Q, ncells, qid: Optional[int] = None):
        cand_beg = time.time()
        cells, scores = self.get_cells_func(Q, ncells, qid=qid)
        # cells, scores = self.get_cells(Q, ncells)
        elapsed_ms = (time.time() - cand_beg) * 1000
        self.tot_time += elapsed_ms
        self.tot_cells += len(cells)
        # self.cells_retrieved.append(cells.tolist())

        pids, cell_lengths = self.ivf.lookup(cells)
        if self.use_gpu:
            pids = pids.cuda()
        return pids, scores

    def generate_candidate_scores(self, Q, eids):
        E = self.lookup_eids(eids)
        if self.use_gpu:
            E = E.cuda()
        return (Q.unsqueeze(0) @ E.unsqueeze(2)).squeeze(-1).T

    def generate_candidates(self, config, Q, qid: Optional[int] = None):
        ncells = config.ncells

        assert isinstance(self.ivf, StridedTensor)

        Q = Q.squeeze(0)
        if self.use_gpu:
            Q = Q.cuda().half()
        assert Q.dim() == 2

        pids, centroid_scores = self.generate_candidate_pids(Q, ncells, qid=qid)

        sorter = pids.sort()
        pids = sorter.values

        pids, pids_counts = torch.unique_consecutive(pids, return_counts=True)
        if self.use_gpu:
            pids, pids_counts = pids.cuda(), pids_counts.cuda()

        return pids, centroid_scores
