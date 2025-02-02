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
    centroid_idx: int,
    pids: torch.Tensor,
    codes: torch.Tensor,
    code_offsets: torch.Tensor,
    doclens: torch.Tensor,
    out_centroid_pids: torch.Tensor,
):
    """Sort pids by their count of embeddings in the cluster."""
    counts = torch.zeros(len(pids), dtype=torch.int32, device="cpu")
    for i, p in enumerate(pids):
        for i in range(doclens[p]):
            curr_code = codes[code_offsets[p] + i]
            if curr_code.item() == centroid_idx:
                counts[i] += 1

    if (counts == 0).any():
        raise Exception("passage with 0 count")

    out_centroid_pids[centroid_idx] = torch.argsort(counts, descending=True)


def sort_similar(
    centroid_idx: int,
    pids: torch.Tensor,
    pids_embs: torch.Tensor,
    pids_lengths: torch.Tensor,
    centroids: torch.Tensor,
    lengths: torch.Tensor,
    out_centroid_pids: torch.Tensor,
):
    """Sort pids by their similarity to centroid."""
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

    k = lengths[centroid_idx].item()
    while j < k:
        next_m = m + k - j
        if next_m >= len(emb_scores.indices):
            next_m = len(emb_scores.indices) - 1

        max_pids = max_pids_all[m:next_m]
        next_j = j
        for h in range(len(max_pids)):
            mp = pids[max_pids[h].item()].item()
            if mp in chosen_set:
                continue

            chosen_set.add(mp)
            cid_pids[centroid_idx, next_j] = mp
            next_j += 1

        j = next_j
        m = next_m


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
        embs_strided.codes_strided.offsets
        n = len(lengths)
        cid_pids = torch.full(
            (n, torch.max(lengths)), -1, dtype=int, device="cpu"
        )
        with tqdm(total=n, desc="Find closest passage to centroid") as pbar:
            for i in range(n):
                pids = all_pids[offsets[i] : offsets[i + 1]]
                if args.order_method == "similar":
                    pids_embs, pids_lengths = embs_strided.lookup_pids(pids)
                    pids_embs = pids_embs.to(device)
                    pids_lengths = pids_lengths.to(device)
                    sort_similar(
                        i,
                        pids,
                        pids_embs,
                        pids_lengths,
                        centroids,
                        lengths,
                        cid_pids,
                    )
                elif args.order_method == "count":
                    sort_count(
                        i,
                        pids,
                        embs_strided.codes,
                        embs_strided.codes_strided.offsets,
                        loader.doclens,
                        cid_pids,
                    )
                else:
                    raise ValueError(f"no such method: {args.order_method}")

                pbar.update(1)

        torch.save(cid_pids, osp.join(index_path, args.out_filename))
