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
            "attention": self.att_prune,
            "first_alpha": self.first_alpha_prune,
        }

        self.static_prune_func = None
        if self.config.static_pruning_method is not None:
            if (
                sm := self.config.static_pruning_method
                not in self.static_prune_funcs
            ):
                raise ValueError(f"No such static pruning method: '{sm}'")

            self.static_prune_func = self.static_prune_funcs[sm]

        self.cache_path = None
        if self.config.encode_cache_path is not None:
            self.cache_path = Path(self.config.encode_cache_path)

    def _constant_prune_num(self, dlen: int, prune_ratio: float) -> int:
        return int(np.ceil(dlen * (1 - prune_ratio)))

    def first_alpha_prune(
        self, embs: torch.Tensor, prune_ratio=0.25
    ) -> torch.Tensor:
        return embs[: self._constant_prune_num(embs.shape[0], prune_ratio)]

    def att_prune(
        self, embs: torch.Tensor, prune_ratio=0.25
    ) -> torch.Tensor:
        p_doc_len = self._constant_prune_num(embs.shape[0], prune_ratio)
        att_sum = torch.sum(embs @ embs.T, dim=0)

        # Keep smallest p_doc_len
        keep_dims = torch.argsort(att_sum)[:p_doc_len]

        # Sort from smallest index to largest to not mix the order
        keep_dims = torch.sort(keep_dims).values

        return embs[keep_dims]

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
