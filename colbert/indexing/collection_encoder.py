from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np

from colbert.infra.run import Run
from colbert.utils.utils import print_message, batch


class CollectionEncoder:

    def __init__(self, config, checkpoint):
        self.config = config
        self.checkpoint = checkpoint
        self.use_gpu = self.config.gpus > 0
        self.static_prune_funcs = {
            "att_top": self.att_top_keep,
            "att_bottom": self.att_bottom_keep,
            "first": self.first_keep,
            "att_greedy_mvc": self.att_greedy_mvc,
        }

        self.static_prune_func = None
        if self.config.static_pruning_method is not None:
            if (
                sm := self.config.static_pruning_method
            ) not in self.static_prune_funcs:
                raise ValueError(f"No such static pruning method: '{sm}'")

            self.static_prune_func = self.static_prune_funcs[sm]

        self.cache_path = None
        if self.config.encode_cache_path is not None:
            self.cache_path = Path(self.config.encode_cache_path)

    def _constant_keep_num(self, dlen: int, keep_ratio: float) -> int:
        return int(np.ceil(dlen * keep_ratio))

    def first_keep(self, embs: torch.Tensor, keep_ratio=0.75) -> torch.Tensor:
        return embs[: self._constant_keep_num(embs.shape[0], keep_ratio)]

    def _att_keep(self, embs: torch.Tensor, keep_ratio=0.75, keep_top=False):
        p_doc_len = self._constant_keep_num(embs.shape[0], keep_ratio)

        # Discard first 2 (CLS and D) from evaluation
        cand_embs = embs[2:, :]
        att_sum = torch.sum(cand_embs @ cand_embs.T, dim=0)

        # Keep smallest p_doc_len - 2
        keep_dims = torch.argsort(att_sum, descending=keep_top)[:p_doc_len - 2]

        # Sort from smallest index to largest to not mix the order
        keep_dims = torch.sort(keep_dims).values

        # Always include CLS and D in the result
        return torch.cat((embs[:2, :], cand_embs[keep_dims]))

    def att_top_keep(self, embs: torch.Tensor, keep_ratio=0.75) -> torch.Tensor:
        return self._att_keep(embs, keep_ratio=keep_ratio, keep_top=True)

    def att_bottom_keep(
        self, embs: torch.Tensor, keep_ratio=0.75
    ) -> torch.Tensor:
        return self._att_keep(embs, keep_ratio=keep_ratio, keep_top=False)

    def att_greedy_mvc(
        self, embs: torch.Tensor, edge_threshold=0.8
    ) -> torch.Tensor:
        adj = (embs @ embs.T) >= edge_threshold
        adj[torch.eye(len(adj), dtype=torch.bool)] = False
        keep_mask = torch.zeros(len(embs), dtype=torch.bool)

        # Always keep vertices without edges
        keep_mask[torch.sum(adj, dim=1) == 0] = True

        # Greedily remove the embedding with lowest degree larger than 0.
        while adj.any():
            degree = torch.sum(adj, dim=1)
            keep_idx = torch.argmax(degree).item()
            keep_mask[keep_idx] = True
            adj[keep_idx, :] = False
            adj[:, keep_idx] = False

        return embs[keep_mask]

    def static_prune(
        self, embs: torch.Tensor, doclens: List[int], **prune_kwargs
    ) -> Tuple[torch.Tensor, List[int]]:
        offset = 0
        pruned_embs = []
        pruned_doclens = []
        for dlen in doclens:
            d_embs = embs[offset : offset + dlen]
            if self.use_gpu:
                d_embs = d_embs.to("cuda")

            offset += dlen
            pd_embs = self.static_prune_func(d_embs, **prune_kwargs)
            # Always accumulate in CPU because most likely GPU mem
            # is not enough.
            pruned_embs.append(pd_embs.to("cpu"))
            pruned_doclens.append(pd_embs.shape[0])

        return torch.cat(pruned_embs), pruned_doclens

    def encode_passages(self, passages, chunk_idx=None):
        Run().print(f"#> Encoding {len(passages)} passages..")

        if len(passages) == 0:
            return None, None

        use_cache = self.cache_path is not None and chunk_idx is not None
        embs_path = (
            self.cache_path / f"embs_{chunk_idx}.pt" if use_cache else None
        )
        dlens_path = (
            self.cache_path / f"dlens_{chunk_idx}.pt" if use_cache else None
        )
        with torch.inference_mode():
            if (
                embs_path is not None
                and embs_path.exists()
                and dlens_path is not None
                and dlens_path.exists()
            ):
                embs = torch.load(embs_path)
                doclens = torch.load(dlens_path)
            else:
                embs, doclens = [], []

                # Batch here to avoid OOM from storing intermediate
                # embeddings on GPU.  Storing on the GPU helps with speed
                # of masking, etc.  But ideally this batching happens
                # internally inside docFromText.
                for passages_batch in batch(passages, self.config.bsize * 50):
                    embs_, doclens_ = self.checkpoint.docFromText(
                        passages_batch,
                        bsize=self.config.bsize,
                        keep_dims="flatten",
                        showprogress=(not self.use_gpu),
                    )
                    embs.append(embs_)
                    doclens.extend(doclens_)

                embs = torch.cat(embs)
                if use_cache:
                    self.cache_path.mkdir(parents=True, exist_ok=True)
                    torch.save(embs, embs_path)
                    torch.save(doclens, dlens_path)

        if self.static_prune_func is not None:
            print_message(
                "Static pruning enabled with "
                f"method '{self.config.static_pruning_method}'."
            )
            return self.static_prune(
                embs, doclens, **self.config.static_pruning_kwargs
            )

        return embs, doclens
