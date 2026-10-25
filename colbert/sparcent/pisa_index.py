"""Index the sparse centroid documents with PISA and convert them to BMP.

The documents are inverted by PISA, reordered by recursive graph bisection,
exported to CIFF and imported into a BMP index.
"""

import gzip
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from colbert.utils.utils import print_message

PISA_TOOLS = ("parse_collection", "invert", "reorder-docids", "lexicon")


def _tool(name: str, bin_dir: Optional[str] = None) -> str:
    path = os.path.join(bin_dir, name) if bin_dir else shutil.which(name)
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(
            f"'{name}' not found, add it to PATH or set sparcent_pisa_bin_dir."
        )

    return path


def check_tools(pisa_bin_dir: Optional[str] = None):
    """Raise if one of the PISA tools or ciff's pisa2ciff is missing."""
    for name in PISA_TOOLS:
        _tool(name, pisa_bin_dir)

    _tool("pisa2ciff")


def _run(cmd: List[str], **kwargs):
    cmd = [str(c) for c in cmd]
    print_message(f"#> Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def _parse_collection(docs_path: Path, cmd: List[str]):
    """Run parse_collection, which reads the documents from stdin."""
    cmd = [str(c) for c in cmd]
    print_message(f"#> Running: zcat {docs_path} | {' '.join(cmd)}")
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as parse:
        with gzip.open(docs_path, "rb") as docs:
            shutil.copyfileobj(docs, parse.stdin)

        parse.stdin.close()
        if parse.wait() != 0:
            raise subprocess.CalledProcessError(parse.returncode, cmd)


def build_bmp_index(
    docs_path: Path,
    work_dir: Path,
    out_path: Path,
    bp_depth: int = 13,
    jobs: int = 8,
    batch_size: int = 32768,
    block_size: int = 8,
    compress_range: bool = True,
    pisa_bin_dir: Optional[str] = None,
):
    """Build the BMP index `out_path` from gzipped `docid<TAB>terms` docs."""
    import bmp

    name = "centroids"
    bp_name = f"{name}_bp"
    fwd_dir = work_dir / "forward"
    inv_dir = work_dir / "inverted"
    fwd_dir.mkdir(parents=True, exist_ok=True)
    inv_dir.mkdir(parents=True, exist_ok=True)

    _parse_collection(
        docs_path,
        [
            _tool("parse_collection", pisa_bin_dir),
            "-o",
            fwd_dir / name,
            "-b",
            batch_size,
            "-j",
            jobs,
            "--tokenizer",
            "whitespace",
        ],
    )
    _run(
        [
            _tool("invert", pisa_bin_dir),
            "-i",
            fwd_dir / name,
            "-o",
            inv_dir / name,
            "-j",
            jobs,
            "--batch-size",
            batch_size,
        ]
    )
    _run(
        [
            _tool("reorder-docids", pisa_bin_dir),
            "-c",
            inv_dir / name,
            "-o",
            inv_dir / bp_name,
            "--documents",
            fwd_dir / f"{name}.doclex",
            "--reordered-documents",
            fwd_dir / f"{bp_name}.doclex",
            "-d",
            bp_depth,
            "-j",
            jobs,
            "--bp",
        ]
    )
    with open(fwd_dir / f"{bp_name}.documents", "w") as documents:
        _run(
            [
                _tool("lexicon", pisa_bin_dir),
                "print",
                fwd_dir / f"{bp_name}.doclex",
            ],
            stdout=documents,
        )

    ciff_path = work_dir / f"{bp_name}.ciff"
    _run(
        [
            _tool("pisa2ciff"),
            "-c",
            inv_dir / bp_name,
            "-d",
            fwd_dir / f"{bp_name}.documents",
            "-t",
            fwd_dir / f"{name}.terms",
            "-o",
            ciff_path,
        ]
    )

    print_message(f"#> Converting {ciff_path} to BMP index {out_path}")
    bmp.ciff2bmp(
        ciff_file=str(ciff_path),
        output=str(out_path),
        bsize=block_size,
        compress_range=compress_range,
    )
