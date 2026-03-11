"""
Multilingual semantic style embeddings.

Uses paraphrase-multilingual-mpnet-base-v2 which supports 50+ languages
and maps semantically similar sentences to the same embedding space
regardless of language.
"""

from __future__ import annotations

import re
from typing import List, Optional
import numpy as np


_model = None
_model_name = "paraphrase-multilingual-mpnet-base-v2"


def _load_model(quiet: bool = False):
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            )
        import os
        import logging
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "idiolect")
        model_path = os.path.join(cache_dir, _model_name)

        if not quiet and not os.path.exists(model_path):
            try:
                from rich.console import Console
                Console(stderr=True).print(
                    "[dim]Downloading multilingual model (~420MB, one-time)...[/dim]"
                )
            except ImportError:
                import sys
                print("Downloading multilingual model (~420MB, one-time)...", file=sys.stderr)

        # Suppress all internal logging noise from transformers / HuggingFace
        for logger_name in ("sentence_transformers", "transformers", "huggingface_hub"):
            logging.getLogger(logger_name).setLevel(logging.ERROR)

        import contextlib, io
        with contextlib.redirect_stderr(io.StringIO()):
            _model = SentenceTransformer(_model_name, cache_folder=cache_dir)
    return _model


def _chunk_text(text: str, max_words: int = 80, overlap: int = 20) -> List[str]:
    """Split text into overlapping chunks for embedding."""
    sentences = re.split(r'(?<=[.!?…])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = []
    current_words = 0

    for sent in sentences:
        word_count = len(sent.split())
        if current_words + word_count > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            # overlap: keep last few sentences
            overlap_sents = []
            overlap_wc = 0
            for s in reversed(current_chunk):
                wc = len(s.split())
                if overlap_wc + wc <= overlap:
                    overlap_sents.insert(0, s)
                    overlap_wc += wc
                else:
                    break
            current_chunk = overlap_sents
            current_words = overlap_wc

        current_chunk.append(sent)
        current_words += word_count

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks if chunks else [text]


def get_style_embedding(text: str, verbose: bool = False, quiet: bool = False) -> np.ndarray:
    """
    Compute a style embedding for the given text.
    Returns a 768-dim vector representing the author's semantic style.
    """
    model = _load_model(quiet=quiet)
    chunks = _chunk_text(text)

    if verbose:
        print(f"  Encoding {len(chunks)} text chunk(s)...")

    embeddings = model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)

    # Weighted average: longer chunks get more weight
    weights = np.array([len(c.split()) for c in chunks], dtype=float)
    weights /= weights.sum()

    style_vec = np.average(embeddings, axis=0, weights=weights)
    norm = np.linalg.norm(style_vec)
    if norm > 0:
        style_vec /= norm

    return style_vec


def embedding_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two style vectors. Returns [-1, 1]."""
    dot = float(np.dot(vec_a, vec_b))
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
