from __future__ import annotations
import itertools
import os
from pathlib import Path
from typing import List, Tuple

import jgrapht
import numpy as np
import torch
from torch.utils.cpp_extension import load


from colbert.infra.run import Run
from colbert.utils.utils import print_message, batch


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
            keep_idx = torch.argmax(degree).item()
            keep_mask[keep_idx] = False
            adj[keep_idx, :] = False
            adj[:, keep_idx] = False

        return embs[keep_mask]

    # def att_greedy_mvc(
    #     self, embs: torch.Tensor, edge_threshold=0.8
    # ) -> torch.Tensor:
    #     adj = (embs @ embs.T) >= edge_threshold
    #     adj[torch.eye(len(adj), dtype=torch.bool)] = False

    #     # Disconnect [CLS] and [D] tokens.
    #     adj[:2, :] = False
    #     adj[:, :2] = False

    #     # Always keep disconnected vertices.
    #     keep_mask = torch.zeros(len(embs), dtype=torch.bool)
    #     keep_mask[torch.sum(adj, dim=1) == 0] = True

    #     doclen = len(adj)
    #     graph = jgrapht.create_graph(directed=False, weighted=False)
    #     graph.add_vertices_from(range(doclen))
    #     for i in range(doclen):
    #         for j in range(i + 1, doclen):
    #             if adj[i][j].item():
    #                 graph.add_edge(i, j)

    #     _, vertex_set = jgrapht.algorithms.vertexcover.greedy(graph)
    #     keep_mask[list(vertex_set)] = True

        # mvc_cand = torch.zeros_like(adj, dtype=torch.bool)
        # emb_ids = range(doclen)
        # found = False
        # print_message(f"Solving MVC for size={doclen}")
        # for mvc_len in range(1, doclen + 1):
        #     print_message(f"Trying with size={mvc_len}")
        #     for idx_s in itertools.combinations(emb_ids, mvc_len):
        #         mvc_cand[idx_s, :] = adj[idx_s, :]
        #         mvc_cand[:, idx_s] = adj[:, idx_s]
        #         if (mvc_cand == adj).all():
        #             found = True
        #             print_message(f"MVC found with size={len(idx_s)}")
        #             keep_mask[idx_s] = True
        #             break
        #         else:
        #             mvc_cand.zero_()

        #     if found:
        #         break

        # if not found:
        #     raise ValueError("could not find MVC!")

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
                embs, doclens, **self.config.static_pruning_kwargs
            )

        return embs, doclens
