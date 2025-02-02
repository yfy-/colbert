#!/usr/bin/env python3

import argparse
import os.path as osp
import torch
from tqdm import tqdm

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.search.index_storage import IndexLoader
from colbert.indexing.codecs.residual_embeddings_strided import (
    ResidualEmbeddingsStrided,
)


def sort_count(
    pids: torch.Tensor,
    doclens: torch.Tensor,
    codes: torch.Tensor,
    code_offsets: torch.Tensor,
    centroid_idx: int,
) -> torch.Tensor:
    """Sort pids by their count of embeddings in the cluster."""
    n = len(pids)
    counts = torch.zeros(n, dtype=torch.int32, device="cpu")
    for i, p in enumerate(pids):
        for j in range(doclens[p]):
            curr_code = codes[code_offsets[p] + j]
            if curr_code.item() == centroid_idx:
                counts[i] += 1

    return torch.argsort(counts, descending=True)


def sort_similar(
    pids: torch.Tensor,
    device: str,
    centroids: torch.Tensor,
    centroid_idx: int,
) -> torch.Tensor:
    """Sort pids by their similarity to centroid."""
    pids_embs, pids_lengths = embs_strided.lookup_pids(pids)
    pids_embs = pids_embs.to(device)
    pids_lengths = pids_lengths.to(device)
    emb_scores = torch.sort(
        (centroids[centroid_idx : centroid_idx + 1, :] @ pids_embs.T).squeeze(),
        descending=True,
    )
    pids_lengths_cumsum = torch.cumsum(pids_lengths, dim=0)
    j = 0
    m = 0
    chosen_set = set()
    max_pids_all = torch.searchsorted(
        pids_lengths_cumsum, emb_scores.indices, right=True
    )

    n = len(pids)
    psgs_sorted = torch.zeros(n, dtype=torch.int32)
    while j < n:
        next_m = m + n - j
        if next_m >= len(emb_scores.indices):
            next_m = len(emb_scores.indices) - 1

        max_pids = max_pids_all[m:next_m]
        next_j = j
        for h in range(len(max_pids)):
            mp = pids[max_pids[h].item()].item()
            if mp in chosen_set:
                continue

            chosen_set.add(mp)
            psgs_sorted[next_j] = mp
            next_j += 1

        j = next_j
        m = next_m

    return psgs_sorted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "order_method",
        choices=["similar", "count"],
        help="Method to output passages in order.",
    )
    parser.add_argument("out_filename", help="Out tensor file name")
    parser.add_argument(
        "-d",
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Pytorch device.",
    )
    args = parser.parse_args()
    device = args.device
    with Run().context(RunConfig(nranks=1)):
        config = ColBERTConfig(experiment="msmarco")
        index_path = osp.join(
            config.root, config.experiment, "indexes", "msmarco.nbits=2"
        )
        loader = IndexLoader(index_path)
        all_pids = loader.ivf.tensor.to(device)
        offsets = loader.ivf.offsets.to(device)
        lengths = loader.ivf.lengths
        centroids = loader.codec.centroids.to(device)
        embs_strided = ResidualEmbeddingsStrided(
            loader.codec, loader.embeddings, loader.doclens
        )
        n = len(lengths)
        cid_pids = []
        with tqdm(total=n, desc="Find closest passage to centroid") as pbar:
            for i in range(n):
                pids = all_pids[offsets[i] : offsets[i + 1]]
                if args.order_method == "similar":
                    pids_sorted = sort_similar(pids, device, centroids, i)
                elif args.order_method == "count":
                    pids_sorted = sort_count(
                        pids,
                        loader.doclens,
                        embs_strided.codes,
                        embs_strided.codes_strided.offsets,
                        i,
                    )
                else:
                    raise ValueError(f"no such method: {args.order_method}")

                cid_pids.append(pids_sorted)
                pbar.update(1)

        torch.save(cid_pids, osp.join(index_path, args.out_filename))
