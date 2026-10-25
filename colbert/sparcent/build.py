"""Build the Sparcent sparse centroid index of a ColBERT index.

All outputs are stored in the `sparcent/` directory of the ColBERT index:
the passage order of each centroid, the SPLADE encodings of the collection,
the sparse centroid documents and the final BMP index.
"""

import dataclasses
import gzip
import importlib.util
import os
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import ujson

from colbert.infra.config import ColBERTConfig
from colbert.infra.config.settings import IndexingSettings
from colbert.sparcent import bmp_index_path, sparcent_path
from colbert.sparcent.pisa_index import build_bmp_index, check_tools
from colbert.sparcent.sample_centroids import sample_centroids
from colbert.utils.utils import print_message


def check_requirements(config: ColBERTConfig):
    """Raise if a package or tool needed for the Sparcent build is missing."""
    missing = [
        m for m in ("splade", "bmp") if importlib.util.find_spec(m) is None
    ]
    if missing:
        raise ImportError(
            f"Building a Sparcent index needs the {' and '.join(missing)} "
            "package(s), install colbert with its 'sparcent' extra."
        )

    check_tools(config.sparcent_pisa_bin_dir)


def _encoding_progress(
    encode_dir: Path, collection: str, config: ColBERTConfig
) -> Tuple[bool, int]:
    """Whether the SPLADE encodings are complete and the first missing batch."""
    from splade.sparcent.centroid_docs import batch_path

    info = {
        "collection": os.path.abspath(collection),
        "model": config.sparcent_splade_model,
        "lora_dir": config.sparcent_splade_lora_dir,
        "batch_size": config.sparcent_splade_batch_size,
    }
    info_path = encode_dir / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            if ujson.load(f) != info:
                raise ValueError(
                    f"{encode_dir} has SPLADE encodings of another collection "
                    "or model, remove it to encode again."
                )
    else:
        encode_dir.mkdir(parents=True, exist_ok=True)
        with open(info_path, "w") as f:
            ujson.dump(info, f, indent=4)

    with open(collection) as f:
        num_batches = int(np.ceil(sum(1 for _ in f) / info["batch_size"]))

    start_batch = 0
    while batch_path(encode_dir, start_batch).exists():
        start_batch += 1

    return start_batch >= num_batches, start_batch


def _save_metadata(config: ColBERTConfig, index_path: str):
    """Record the Sparcent indexing parameters in the index metadata."""
    metadata_path = os.path.join(index_path, "metadata.json")
    with open(metadata_path) as f:
        metadata = ujson.load(f)

    for field in dataclasses.fields(IndexingSettings):
        if field.name.startswith("sparcent"):
            metadata["config"][field.name] = getattr(config, field.name)

    with open(metadata_path, "w") as f:
        ujson.dump(metadata, f, indent=4)


def build_sparcent_index(config: ColBERTConfig, index_path: str, collection):
    """Build `sparcent/centroids.bmp` for the ColBERT index at `index_path`."""
    from splade.sparcent.centroid_docs import create_centroid_docs

    if not isinstance(collection, str):
        collection = collection.provenance()

    out_dir = Path(sparcent_path(index_path))
    out_dir.mkdir(exist_ok=True)

    # 1. Order the passages of each centroid.
    method = config.sparcent_sample_method
    sample_path = out_dir / f"sample_pids_{method}.pt"
    if sample_path.exists():
        print_message(f"#> Reusing the centroid passage order {sample_path}")
    else:
        sample_centroids(
            index_path,
            method,
            str(sample_path),
            device=config.sparcent_sample_device,
            jobs=config.sparcent_sample_jobs,
        )

    # 2. Sparse centroid vectors as documents of repeated terms.
    encode_dir = out_dir / "splade_encodings"
    skip_encode, encode_batch = _encoding_progress(
        encode_dir, collection, config
    )
    if not skip_encode:
        # Composed vectors of earlier encodings are stale.
        for compose_dir in out_dir.glob("compose_*"):
            shutil.rmtree(compose_dir)

    lora_dir = config.sparcent_splade_lora_dir
    docs_path = out_dir / "centroid_docs.tsv.gz"
    print_message(f"#> Creating the sparse centroid documents {docs_path}")
    with gzip.open(docs_path, "wt") as docs:
        create_centroid_docs(
            collection,
            encode_dir,
            config.sparcent_splade_model,
            out_dir / f"compose_{method}_n{config.sparcent_nearest_percent}",
            docs,
            batch_size=config.sparcent_splade_batch_size,
            encode_batch_size=config.sparcent_splade_encode_batch_size,
            jobs=config.sparcent_docs_jobs,
            skip_encode=skip_encode,
            encode_batch=encode_batch,
            lora_model_dir=Path(lora_dir) if lora_dir else None,
            doc_compose=str(sample_path),
            nearest_percent=config.sparcent_nearest_percent,
            pruning_submass=config.sparcent_pruning_submass,
        )

    # 3. Inverted index reordered by graph bisection, converted to BMP.
    pisa_dir = out_dir / "pisa"
    shutil.rmtree(pisa_dir, ignore_errors=True)
    build_bmp_index(
        docs_path,
        pisa_dir,
        Path(bmp_index_path(index_path)),
        bp_depth=config.sparcent_bp_depth,
        jobs=config.sparcent_pisa_jobs,
        batch_size=config.sparcent_pisa_batch_size,
        block_size=config.sparcent_bmp_block_size,
        compress_range=config.sparcent_bmp_compress_range,
        pisa_bin_dir=config.sparcent_pisa_bin_dir,
    )

    _save_metadata(config, index_path)
    print_message(f"#> Sparcent index saved to {bmp_index_path(index_path)}")
