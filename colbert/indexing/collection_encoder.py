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

    def static_prune(
        self, embs: torch.Tensor, doclens: List[int]
    ) -> Tuple[torch.Tensor, List[int]]:
        offset = 0
        pruned_embs = []
        pruned_doclens = []
        for dlen in doclens:
            p_doc_len = int(
                np.ceil(dlen * (1 - self.config.static_prune_ratio))
            )
            pruned_doclens.append(p_doc_len)
            d_embs = embs[offset : offset + dlen]
            if self.use_gpu:
                d_embs = d_embs.to("cuda")

            offset += dlen
            att_sum = torch.sum(d_embs @ d_embs.T, dim=0)

            # Keep smallest p_doc_len
            keep_dims = torch.argsort(att_sum)[:p_doc_len]

            # Sort from smallest index to largest to not mix the order
            keep_dims = torch.sort(keep_dims).values

            # Always accumulate in CPU because most likely GPU mem
            # is not enough.
            pruned_embs.append(d_embs[keep_dims].to("cpu"))

        return torch.cat(pruned_embs), pruned_doclens

    def encode_passages(self, passages):
        Run().print(f"#> Encoding {len(passages)} passages..")

        if len(passages) == 0:
            return None, None

        with torch.inference_mode():
            embs, doclens = [], []

            # Batch here to avoid OOM from storing intermediate embeddings on GPU.
            # Storing on the GPU helps with speed of masking, etc.
            # But ideally this batching happens internally inside docFromText.
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

            # embs, doclens = self.checkpoint.docFromText(passages, bsize=self.config.bsize,
            #                                                   keep_dims='flatten', showprogress=(self.config.rank < 1))

        # with torch.inference_mode():
        #     embs = self.checkpoint.docFromText(passages, bsize=self.config.bsize,
        #                                        keep_dims=False, showprogress=(self.config.rank < 1))
        #     assert type(embs) is list
        #     assert len(embs) == len(passages)

        #     doclens = [d.size(0) for d in embs]
        #     embs = torch.cat(embs)

        if self.config.static_prune_ratio:
            print_message(
                "Static pruning enabled with "
                f"alpha={self.config.static_prune_ratio}."
            )
            return self.static_prune(embs, doclens)

        return embs, doclens
