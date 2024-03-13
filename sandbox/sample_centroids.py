#!/usr/bin/env python3

import argparse
import os.path as osp
import torch
from tqdm import tqdm
import logging

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.search.index_storage import IndexLoader
from colbert.indexing.codecs.residual_embeddings_strided import (
    ResidualEmbeddingsStrided,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "num_passage",
        type=int,
        help="Number of passages to sample per centroid.",
    )
    parser.add_argument(
        "-d",
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Pytorch device.",
    )
    args = parser.parse_args()
    k = args.num_passage
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
        cid_pids = torch.full((n, k), -1, dtype=int, device=device)
        num_missing = 0
        with tqdm(total=n, desc="Find closest passage to centroid") as pbar:
            for i in range(n):
                pids = all_pids[offsets[i] : offsets[i + 1]]
                pids_embs, pids_lengths = embs_strided.lookup_pids(pids)
                pids_embs = pids_embs.to(device)
                pids_lengths = pids_lengths.to(device)
                emb_scores = torch.sort(
                    (centroids[i : i + 1, :] @ pids_embs.T).squeeze(),
                    descending=True,
                )
                # max_emb_idx = torch.topk(
                #     (centroids[i:i + 1, :] @ pids_embs.T).squeeze(), k
                # ).indices
                pids_lengths_cumsum = torch.cumsum(pids_lengths, dim=0)
                j = 0
                m = 0
                chosen_set = set()
                max_pids_all = torch.searchsorted(
                    pids_lengths_cumsum, emb_scores.indices, right=True
                )
                # cid_pids[i] = pids[max_pids_all]
                while j < k:
                    next_m = m + k - j
                    if next_m >= len(emb_scores.indices):
                        next_m = len(emb_scores.indices) - 1

                    max_pids = max_pids_all[m:next_m]
                    if len(max_pids) == 0:
                        num_missing += 1
                        logging.warning(
                            f"Only retrieved {len(chosen_set)}/{k} for "
                            f"centroid: '{i}'"
                        )
                        break

                    next_j = j
                    for h in range(len(max_pids)):
                        mp = pids[max_pids[h].item()].item()
                        if mp in chosen_set:
                            continue

                        chosen_set.add(mp)
                        cid_pids[i, next_j] = mp
                        next_j += 1

                    j = next_j
                    m = next_m

                pbar.update(1)

        if num_missing:
            logging.warning(
                f"{num_missing}/{n} centroids do not have '{k}' passages"
            )
        torch.save(cid_pids, osp.join(index_path, f"sample_pids_unq_{k}.pt"))
