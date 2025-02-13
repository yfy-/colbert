#!/usr/bin/env python3

import argparse
import multiprocessing
import os.path as osp
from functools import partial
from typing import Tuple

import torch
from tqdm import tqdm

from colbert.indexing.codecs.residual_embeddings_strided import \
    ResidualEmbeddingsStrided
from colbert.infra import ColBERTConfig, Run, RunConfig
from colbert.search.index_storage import IndexLoader


def sort_count(
    pids_centroid_idx: Tuple[torch.Tensor, int],
    doclens: torch.Tensor,
    codes: torch.Tensor,
    code_offsets: torch.Tensor,
) -> torch.Tensor:
    """Sort pids by their count of embeddings in the cluster."""
    pids, centroid_idx = pids_centroid_idx
    n = len(pids)
    counts = torch.zeros(n, dtype=torch.int32, device="cpu")
    for i, p in enumerate(pids):
        for j in range(doclens[p]):
            curr_code = codes[code_offsets[p] + j]
            if curr_code.item() == centroid_idx:
                counts[i] += 1

    return pids[torch.argsort(counts, descending=True)]


def sort_similar(
    pids_centroid_idx: Tuple[torch.Tensor, int],
    device: str,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """Sort pids by their similarity to centroid."""
    pids, centroid_idx = pids_centroid_idx
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


def sort_mean_similar(
    pids_centroid_idx: Tuple[torch.Tensor, int],
    device: str,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """Sort pids by their mean similarity to the centroid."""
    pids, centroid_idx = pids_centroid_idx
    pids_embs, pids_lengths = embs_strided.lookup_pids(pids)
    pids_embs = pids_embs.to(device)
    pids_lengths = pids_lengths.to(device)
    emb_scores = (
        centroids[centroid_idx : centroid_idx + 1, :] @ pids_embs.T
    ).squeeze(0)
    pids_lengths_cumsum = torch.cumsum(pids_lengths, dim=0)
    max_pids_all = torch.searchsorted(
        pids_lengths_cumsum, torch.arange(len(emb_scores), device=device)
    ).to("cpu")
    emb_scores = emb_scores.to("cpu").double()
    scores = torch.zeros(len(pids), dtype=torch.float64, device="cpu")
    scores.index_add_(0, max_pids_all, emb_scores)
    return pids[torch.argsort(scores / pids_lengths.to("cpu"), descending=True)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "order_method",
        choices=["similar", "count", "mean_similar"],
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
    parser.add_argument("-j", "--jobs", type=int, default=8)
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
        if args.order_method == "similar":
            sort_partial = partial(
                sort_similar, device=device, centroids=centroids
            )
        elif args.order_method == "count":
            sort_partial = partial(
                sort_count,
                doclens=loader.doclens,
                codes=embs_strided.codes,
                code_offsets=embs_strided.codes_strided.offsets,
            )
        elif args.order_method == "mean_similar":
            sort_partial = partial(
                sort_mean_similar, device=device, centroids=centroids
            )
        else:
            raise ValueError(f"no such method: {args.order_method}")

        rest_args = [
            (all_pids[offsets[i] : offsets[i + 1]], i) for i in range(n)
        ]

        if args.jobs == 1:
            cid_pids = list(
                tqdm(
                    map(sort_partial, rest_args),
                    total=n,
                    desc=f"Sorting centroid passages by {args.order_method}",
                )
            )
        else:
            with multiprocessing.Pool(args.jobs) as p:
                cid_pids = list(
                    tqdm(
                        p.imap(sort_partial, rest_args),
                        total=n,
                        desc="Sorting centroid passages by "
                        f"{args.order_method}",
                    )
                )

        torch.save(
            torch.nn.utils.rnn.pad_sequence(
                cid_pids, batch_first=True, padding_value=-1.0
            ),
            osp.join(index_path, args.out_filename),
        )
