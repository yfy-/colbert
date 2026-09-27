from __future__ import annotations

import itertools
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import embpurisma
import numpy as np
import torch
import torch.nn.functional as torch_f
from torch.utils.cpp_extension import load
from tqdm import tqdm

from colbert.data import Queries
from colbert.infra.run import Run
from colbert.utils.utils import batch, print_message


class CollectionEncoder:

    static_prune_query_methods = {"dominant", "dominant_repair", "gas"}
    static_prune_batch_methods = {"gdmis_rs"}

    def __init__(self, config, checkpoint):
        self.config = config
        self.checkpoint = checkpoint
        self.use_gpu = self.config.gpus > 0
        # self.try_load_torch_extensions(self.config.static_pruning_method)

        self.static_prune_funcs = {
            "attention": self.attention_pruning,
            "dissimilar": self.dissimilar_pruning,
            "dominant": self.dominant_pruning,
            "dominant_repair": self.dominant_repair_pruning,
            "first": self.first_pruning,
            "gdmis": self.gdmis_pruning,
            "gdmis-rs": self.gdmis_rs_pruning,
            "similar": self.similar_pruning,
            "gas": self.gas,
            "idf": self.idf_pruning,
        }

        self.static_prune_func = None
        if self.config.static_pruning_method is not None:
            if (
                sm := self.config.static_pruning_method
            ) not in self.static_prune_funcs:
                raise ValueError(f"No such static pruning method: '{sm}'")

            self.static_prune_func = self.static_prune_funcs[sm]
            self.static_pruning_kwargs = dict(self.config.static_pruning_kwargs)
            if sm in self.static_prune_query_methods:
                self.checkpoint.query_tokenizer.query_maxlen = (
                    self.config.query_maxlen
                )

                queries = Queries(self.static_pruning_kwargs["queries"])
                query_texts = list(queries.values())
                bsize = 128 if len(queries) > 128 else None
                print_message(f"Encoding queries for '{sm}'")
                Q = self.checkpoint.queryFromText(
                    query_texts,
                    bsize=bsize,
                    # Accumulate the entire query log in CPU since it can be very large.
                    to_cpu=False,
                    full_length_search=False,
                )
                Q = Q.view(Q.shape[0] * Q.shape[1], Q.shape[2])
                if self.use_gpu:
                    Q = Q.half()

                self.static_pruning_kwargs["queries"] = Q
            if sm == "idf":
                dfs = torch.load(self.static_pruning_kwargs["dfs"])
                lowest_idfs = self.compute_lowest_idf(dfs, 100).to(
                    "cuda" if self.use_gpu else "cpu"
                )
                del self.static_pruning_kwargs["dfs"]
                self.static_pruning_kwargs["lowest_idfs"] = lowest_idfs

        self.cache_path = None
        if self.config.encode_cache_path is not None:
            self.cache_path = Path(self.config.encode_cache_path)

    @staticmethod
    def compute_lowest_idf(dfs: torch.Tensor, k: int) -> torch.Tensor:
        dfs = dfs.clone()
        # 3 is for not including special token IDS: [CLS] (102), [D] (101), [SEP] (2)
        spec_tokens = torch.tensor([2, 101, 102])
        dfs[spec_tokens] = -1
        return torch.topk(dfs, k).indices

    @staticmethod
    def _constant_keep_num(dlen: int, keep_ratio: float) -> int:
        return int(np.ceil(dlen * keep_ratio))

    def attention_pruning(
        self,
        pid: int,
        embs: torch.Tensor,
        tokens: torch.Tensor,
        keep_ratio=0.75,
    ) -> torch.Tensor:
        _, _ = pid, tokens
        p_doc_len = self._constant_keep_num(embs.shape[0], keep_ratio)

        # Discard first 2 (CLS and D) from evaluation
        cand_embs = embs[2:, :]
        sim = cand_embs @ cand_embs.T

        # Set diagonal to prevent self attention dominating.
        sim[torch.eye(len(sim), dtype=torch.bool, device=sim.device)] = float(
            "-inf"
        )

        att = torch.sum(torch.nn.functional.softmax(sim, dim=0), dim=1)

        keep_cand_len = max(0, p_doc_len - 2)
        keep_dims = torch.argsort(att, descending=True)[:keep_cand_len]

        keep_dims = torch.sort(keep_dims).values

        return torch.cat((embs[:2, :], cand_embs[keep_dims]))

    def first_pruning(
        self,
        pid: int,
        embs: torch.Tensor,
        tokens: torch.Tensor,
        keep_ratio=0.75,
    ) -> torch.Tensor:
        _, _ = pid, tokens
        return embs[: self._constant_keep_num(embs.shape[0], keep_ratio)]

    def _similar_keep(
        self, embs: torch.Tensor, keep_ratio=0.75, keep_top=False
    ):
        p_doc_len = self._constant_keep_num(embs.shape[0], keep_ratio)

        # Discard first 2 (CLS and D) from evaluation
        cand_embs = embs[2:, :]
        sim_sum = torch.sum(cand_embs @ cand_embs.T, dim=0)

        # Keep smallest p_doc_len - 2
        keep_cand_len = max(0, p_doc_len - 2)
        keep_dims = torch.argsort(sim_sum, descending=keep_top)[:keep_cand_len]

        # Sort from smallest index to largest to not mix the order
        keep_dims = torch.sort(keep_dims).values

        # Always include CLS and D in the result
        return torch.cat((embs[:2, :], cand_embs[keep_dims]))

    def similar_pruning(
        self,
        pid: int,
        embs: torch.Tensor,
        tokens: torch.Tensor,
        keep_ratio=0.75,
    ) -> torch.Tensor:
        _, _ = pid, tokens
        return self._similar_keep(embs, keep_ratio=keep_ratio, keep_top=True)

    def dissimilar_pruning(
        self,
        pid: int,
        embs: torch.Tensor,
        tokens: torch.Tensor,
        keep_ratio=0.75,
    ) -> torch.Tensor:
        _, _ = pid, tokens
        return self._similar_keep(embs, keep_ratio=keep_ratio, keep_top=False)

    def gdmis_pruning(
        self,
        pid: int,
        embs: torch.Tensor,
        tokens: torch.Tensor,
        edge_threshold=0.8,
    ) -> torch.Tensor:
        _, _ = pid, tokens
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

    def idf_pruning(
        self,
        pid: int,
        embs: torch.Tensor,
        tokens: torch.Tensor,
        lowest_idfs: torch.Tensor,
    ):
        _ = pid
        keep_mask = torch.isin(tokens, lowest_idfs, invert=True)
        return embs[keep_mask]

    @classmethod
    def _dominance_probs(
        cls, embs: torch.Tensor, queries: torch
    ) -> torch.Tensor:
        # Q x N
        scores = queries @ embs.T
        max_emb_args = torch.argmax(scores, dim=1)
        counts = torch.bincount(max_emb_args, minlength=embs.shape[0])
        return counts / queries.shape[0]

    @classmethod
    def _arg_most_dominant(
        cls, embs: torch.Tensor, queries: torch.Tensor, dominance_ratio=0.01
    ) -> torch.Tensor:
        return cls._dominance_probs(embs, queries) >= dominance_ratio

    @classmethod
    def gas(
        cls,
        pid: int,
        embs: torch.Tensor,
        tokens: torch.Tensor,
        queries: torch.Tensor,
        keep_ratio: float = 0.75,
        c: float = 1.0,
    ) -> torch.Tensor:
        _, _ = pid, tokens
        importances = cls._dominance_probs(embs, queries)
        keep_mask = torch.zeros(
            embs.shape[0], dtype=torch.bool, device=embs.device
        )
        penalty_sims = (embs @ embs.T) * (2.0 * c)

        for _ in range(cls._constant_keep_num(embs.shape[0], keep_ratio)):
            most_important = torch.argmax(importances)
            keep_mask[most_important] = True
            importances[most_important] = float("-inf")
            importances -= penalty_sims[most_important]

        return embs[keep_mask]

    @classmethod
    def dominant_pruning(
        cls,
        pid: int,
        embs: torch.Tensor,
        tokens: torch.Tensor,
        queries: torch.Tensor,
        dominance_ratio=0.01,
    ) -> torch.Tensor:
        _, _ = pid, tokens
        return embs[
            cls._arg_most_dominant(
                embs,
                queries,
                dominance_ratio=dominance_ratio,
            )
        ]

    @classmethod
    def dominant_repair_pruning(
        cls,
        pid: int,
        embs: torch.Tensor,
        tokens: torch.Tensor,
        queries: torch.Tensor,
        dominance_ratio=0.01,
        iters=3,
        num_neighbors=5,
        step_size=0.25,
    ) -> torch.Tensor:
        _, _ = pid, tokens
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

            # Move in the direction of spherical mean by stepsize.
            kept_embs[nbrs[nbr_mask]] = torch_f.normalize(
                kept_embs[nbrs[nbr_mask]] * (1 - step_size)
                + torch_f.normalize(code_sums[nbr_mask], dim=1) * step_size,
                dim=1,
            )

        return kept_embs

    def gdmis_rs_pruning(
        self, embs: torch.Tensor, doclens: List[int], edge_threshold=0.8
    ):
        embs = embs.to("cpu")
        len_cumsum = np.cumsum([0] + doclens)
        keep_mask = torch.ones(len_cumsum[-1], dtype=bool)
        gdmis_doclens = []
        for offset, dlen in zip(len_cumsum[:-1], doclens):
            keep_mask[offset : offset + 2] = False
            gdmis_doclens.append(dlen - 2)

        pruned_embs, pruned_doclens = embpurisma.gdmis_batch(
            embs[keep_mask].detach().cpu().numpy(),
            gdmis_doclens,
            edge_threshold,
        )
        pruned_embs = torch.from_numpy(pruned_embs)

        final_embs = torch.empty(
            len(doclens) * 2 + pruned_embs.shape[0], dtype=embs.dtype
        )
        final_doclens = []
        offset = 0
        p_offset = 0
        for i, dlen, p_dlen in enumerate(zip(doclens, pruned_doclens)):
            f_offset = p_offset + i * 2
            final_embs[f_offset : f_offset + 2] = embs[offset : offset + 2]

            final_embs[f_offset + 2 : f_offset + 2 + p_dlen] = pruned_embs[
                p_offset : p_offset + p_dlen
            ]
            final_doclens.append(p_dlen + 2)
            offset += dlen
            p_offset += p_dlen

        if self.use_gpu:
            final_embs = final_embs.to("cuda")

        return final_embs, final_doclens

    def static_prune_iterative(
        self,
        embs: torch.Tensor,
        doclens: List[int],
        tokens: torch.Tensor,
        pids: List[int],
        **prune_kwargs,
    ) -> Tuple[torch.Tensor, List[int]]:
        offset = 0
        pruned_embs = []
        pruned_doclens = []
        for (
            pid,
            dlen,
        ) in tqdm(
            iterable=zip(pids, doclens),
            total=len(pids),
            desc="Static pruning",
        ):
            d_embs = embs[offset : offset + dlen]
            d_toks = tokens[offset : offset + dlen]
            if self.use_gpu:
                d_embs = d_embs.to("cuda")
                d_toks = d_toks.to("cuda")

            offset += dlen
            pd_embs = self.static_prune_func(
                pid, d_embs, d_toks, **prune_kwargs
            )
            # Always accumulate in CPU because most likely GPU mem
            # is not enough.
            pruned_embs.append(pd_embs.to("cpu"))
            pruned_doclens.append(pd_embs.shape[0])

        print(f"Avg doclen after pruning: {np.mean(pruned_doclens)}")
        return torch.cat(pruned_embs), pruned_doclens

    # TODO: Most likely we need to pass passage ids separately to this
    # function to support most dominant pruning method.
    def encode_passages(
        self,
        passages: List[str],
        chunk_idx: Optional[int] = None,
        pids: Optional[List[int]] = None,
    ):
        Run().print(f"#> Encoding {len(passages)} passages..")

        if len(passages) == 0:
            return None, None

        if self.static_prune_func is not None and pids is None:
            raise ValueError("Static pruning cannot run without pids!")

        use_cache = self.cache_path is not None and chunk_idx is not None
        embs_path = (
            self.cache_path / f"embs_{chunk_idx}.pt" if use_cache else None
        )
        dlens_path = (
            self.cache_path / f"dlens_{chunk_idx}.pt" if use_cache else None
        )
        tokens_path = (
            self.cache_path / f"tokens_{chunk_idx}.pt" if use_cache else None
        )
        with torch.inference_mode():
            if (
                embs_path is not None
                and embs_path.exists()
                and dlens_path is not None
                and dlens_path.exists()
                and tokens_path is not None
                and tokens_path.exists()
            ):
                embs = torch.load(embs_path)
                doclens = torch.load(dlens_path)
                tokens = torch.load(tokens_path)
            else:
                embs, doclens, tokens = [], [], []

                # Batch here to avoid OOM from storing intermediate
                # embeddings on GPU.  Storing on the GPU helps with speed
                # of masking, etc.  But ideally this batching happens
                # internally inside docFromText.
                for passages_batch in batch(passages, self.config.bsize * 50):
                    embs_, doclens_, tokens_ = self.checkpoint.docFromText(
                        passages_batch,
                        bsize=self.config.bsize,
                        keep_dims="flatten",
                        showprogress=(not self.use_gpu),
                        return_tokens=True,
                    )
                    embs.append(embs_)
                    doclens.extend(doclens_)
                    tokens.append(tokens_)

                embs = torch.cat(embs)
                tokens = torch.cat(tokens)
                if use_cache:
                    self.cache_path.mkdir(parents=True, exist_ok=True)
                    torch.save(embs, embs_path)
                    torch.save(doclens, dlens_path)
                    torch.save(tokens, tokens_path)

        if self.static_prune_func is not None:
            print_message(
                "Static pruning enabled with "
                f"method '{self.config.static_pruning_method}'."
            )

            if self.config.static_pruning_method in self.static_prune_batch_methods:
                return self.static_prune_func(
                    embs, doclens, **self.static_pruning_kwargs
                )

            return self.static_prune_iterative(
                embs, doclens, tokens, pids, **self.static_pruning_kwargs
            )

        return embs, doclens
