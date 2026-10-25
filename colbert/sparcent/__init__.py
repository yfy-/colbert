"""Sparcent: sparse centroid indices for PLAID candidate generation."""

import os

SPARCENT_DIR = "sparcent"
BMP_INDEX = "centroids.bmp"


def sparcent_path(index_path: str) -> str:
    return os.path.join(index_path, SPARCENT_DIR)


def bmp_index_path(index_path: str) -> str:
    return os.path.join(sparcent_path(index_path), BMP_INDEX)
