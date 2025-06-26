from __future__ import annotations

import itertools
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as torch_f
from torch.utils.cpp_extension import load

from colbert.data import Queries
from colbert.infra.run import Run
from colbert.utils.utils import batch, print_message


class CollectionEncoder:

    def __init__(self, config, checkpoint):
        self.config = config
        self.checkpoint = checkpoint
        self.use_gpu = self.config.gpus > 0
        # self.try_load_torch_extensions(self.config.static_pruning_method)

        self.static_prune_funcs = {
            "att_top": self.att_top_keep,
            "att_bottom": self.att_bottom_keep,
            "first": self.first_keep,
            "att_greedy_mvc": self.att_greedy_mvc,
            "most_dominant": self.most_dominant,
            "most_dominant_repair": self.most_dominant_repair,
        }

        self.static_prune_func = None
        if self.config.static_pruning_method is not None:
            if (
                sm := self.config.static_pruning_method
            ) not in self.static_prune_funcs:
                raise ValueError(f"No such static pruning method: '{sm}'")

            self.static_prune_func = self.static_prune_funcs[sm]
            self.static_pruning_kwargs = dict(self.config.static_pruning_kwargs)
            if self.config.static_pruning_method.startswith("most_dominant"):
                self.checkpoint.query_tokenizer.query_maxlen = (
                    self.config.query_maxlen
                )

                queries = list(
                    Queries(self.static_pruning_kwargs["queries"]).values()
                )
                bsize = 128 if len(queries) > 128 else None
                print_message("Encoding queries for 'most_dominant'")
                Q = self.checkpoint.queryFromText(
                    queries,
                    bsize=bsize,
                    to_cpu=not self.use_gpu,
                    full_length_search=False,
                )
                Q = Q.view(Q.shape[0] * Q.shape[1], Q.shape[2])
                if self.use_gpu:
                    Q = Q.half()

                self.static_pruning_kwargs["queries"] = Q

        self.cache_path = None
        if self.config.encode_cache_path is not None:
            self.cache_path = Path(self.config.encode_cache_path)

    @classmethod
    def try_load_torch_extensions(cls, prune_method: str):
        if (
            hasattr(cls, "loaded_extensions")
            or prune_method != "att_exhaustive_mvc"
        ):
            return

        print_message(
            "Loading att_exhaustive_mvc extension "
            "(set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)..."
        )
        att_exhaustive_mvc_cpp = load(
            name="att_exhaustive_mvc_cpp",
            sources=[
                str(Path(__file__).parent.resolve() / "att_exhaustive_mvc.cpp")
            ],
            extra_cflags=["-O3"],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False")
            == "True",
        )
        cls.att_exhaustive_mvc = att_exhaustive_mvc_cpp.att_exhaustive_mvc_cpp

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
        keep_dims = torch.argsort(att_sum, descending=keep_top)[: p_doc_len - 2]

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

    # TODO: Benchmark greedy mvc between jgrapht, PyTorch and cpp.
    def att_greedy_mvc(
        self, embs: torch.Tensor, edge_threshold=0.8
    ) -> torch.Tensor:
        adj = (embs @ embs.T) >= edge_threshold

        # Remove self loops.
        adj[torch.eye(len(adj), dtype=torch.bool)] = False

        # Disconnect [CLS] and [D] tokens.
        adj[:2, :] = False
        adj[:, :2] = False

        keep_mask = torch.ones(len(embs), dtype=torch.bool)

        # Always keep vertices without edges
        # keep_mask[torch.sum(adj, dim=1) == 0] = True

        # Greedily remove the embedding with highest degree.
        while adj.any():
            degree = torch.sum(adj, dim=1)
            # Add an increasing small number to larger indices so that
            # ties are broken in favor of smaller indices.
            small_diff = (
                torch.arange(len(adj), device="cuda" if self.use_gpu else None)
                * 1e-6
            )
            keep_idx = torch.argmax(degree + small_diff).item()
            keep_mask[keep_idx] = False
            adj[keep_idx, :] = False
            adj[:, keep_idx] = False

        return embs[keep_mask]

    @classmethod
    def _arg_most_dominant(
        cls, embs: torch.Tensor, queries: torch.Tensor, dominance_ratio=0.01
    ) -> torch.Tensor:
        # Q x N
        scores = queries @ embs.T
        max_emb_args = torch.argmax(scores, dim=1)
        counts = torch.bincount(max_emb_args, minlength=embs.shape[0])
        probs = counts / queries.shape[0]
        return probs >= dominance_ratio

    @classmethod
    def most_dominant(
        cls,
        embs: torch.Tensor,
        queries: torch.Tensor,
        dominance_ratio=0.01,
    ) -> torch.Tensor:
        return embs[
            cls._arg_most_dominant(
                embs, queries, dominance_ratio=dominance_ratio
            )
        ]

    @classmethod
    def most_dominant_repair(
        cls,
        embs: torch.Tensor,
        queries: torch.Tensor,
        dominance_ratio=0.01,
        iters=3,
        num_neighbors=5,
        step_size=0.25,
    ) -> torch.Tensor:
        keep_mask = cls._arg_most_dominant(
            embs, queries, dominance_ratio=dominance_ratio
        )
        if keep_mask.all():
            return embs

        kept_embs = embs[keep_mask]
        rm_embs = embs[~keep_mask]
        nbr_scores = rm_embs @ kept_embs.T
        _, nbrs = torch.topk(
            nbr_scores,
            min(num_neighbors, kept_embs.size(0)),
            dim=1,
            sorted=False,
        )
        nbrs = torch.unique(nbrs, sorted=False)
        for _ in range(iters):
            # Q
            q_codes = torch.argmax(queries @ kept_embs.T, dim=1)

            # Q x N
            q_codes = torch_f.one_hot(q_codes, num_classes=kept_embs.size(0))[
                :, nbrs
            ].to(queries.dtype)

            # N x d
            code_sums = q_codes.T @ queries

            # Only keep nbrs that are assigned at least 1 query.
            nbr_mask = torch.sum(q_codes, dim=0) > 0

            kept_embs[nbrs[nbr_mask]] = (
                kept_embs[nbrs[nbr_mask]] * (1 - step_size)
                + torch_f.normalize(code_sums[nbr_mask], dim=1) * step_size
            )
            torch_f.normalize(
                kept_embs[nbrs[nbr_mask]], dim=1, out=kept_embs[nbrs[nbr_mask]]
            )

        return kept_embs

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

        print(f"Avg doclen after pruning: {np.mean(pruned_doclens)}")
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
                embs, doclens, **self.static_pruning_kwargs
            )

        return embs, doclens
