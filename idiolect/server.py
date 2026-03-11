"""
idiolect REST API server.

Provides HTTP endpoints for programmatic access.
Start with: idiolect serve --port 8080

Requires: pip install fastapi uvicorn
"""

from __future__ import annotations

from typing import List, Optional
from dataclasses import asdict


def create_app():
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "fastapi and uvicorn are required for serve mode:\n"
            "  pip install fastapi uvicorn"
        )

    from .scorer import compare, compare_with_ci, AttributionResult
    from .features import extract_features

    app = FastAPI(
        title="idiolect API",
        description="Cross-lingual author attribution",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class CompareRequest(BaseModel):
        text_a: str
        text_b: str
        use_embeddings: bool = True
        bootstrap: bool = False
        bootstrap_n: int = 100

    class MultiRequest(BaseModel):
        author_sample: str
        unknowns: List[str]
        use_embeddings: bool = True
        labels: Optional[List[str]] = None

    class MatrixRequest(BaseModel):
        texts: List[str]
        labels: Optional[List[str]] = None
        use_embeddings: bool = False   # default off for matrix — can be large

    class FeaturesRequest(BaseModel):
        text: str

    def _result_to_dict(r: AttributionResult, label_a: str = "text_a", label_b: str = "text_b") -> dict:
        return {
            "label_a": label_a,
            "label_b": label_b,
            "lang_a": r.text_a_lang,
            "lang_b": r.text_b_lang,
            "score": r.score,
            "verdict": r.verdict,
            "confidence": r.confidence,
            "feature_score": round(r.feature_score * 100, 1),
            "embedding_score": round(r.embedding_score * 100, 1),
            "ci_low": r.ci_low,
            "ci_high": r.ci_high,
            "top_matches": [
                {"feature": n, "val_a": round(a, 4), "val_b": round(b, 4), "similarity": round(s, 4)}
                for n, a, b, s in r.top_matches
            ],
            "top_mismatches": [
                {"feature": n, "val_a": round(a, 4), "val_b": round(b, 4), "similarity": round(s, 4)}
                for n, a, b, s in r.top_mismatches
            ],
            "warnings": r.warnings,
        }

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "0.1.0"}

    @app.post("/compare")
    def api_compare(req: CompareRequest):
        """Compare two texts for common authorship."""
        if len(req.text_a.strip()) < 50 or len(req.text_b.strip()) < 50:
            raise HTTPException(status_code=400, detail="Each text must be at least 50 characters.")
        try:
            if req.bootstrap:
                result = compare_with_ci(req.text_a, req.text_b, n_bootstrap=req.bootstrap_n)
            else:
                result = compare(req.text_a, req.text_b, use_embeddings=req.use_embeddings, quiet=True)
            return _result_to_dict(result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/multi")
    def api_multi(req: MultiRequest):
        """Compare one known text against multiple unknowns, ranked by score."""
        if not req.unknowns:
            raise HTTPException(status_code=400, detail="Provide at least one unknown text.")
        labels = req.labels or [f"unknown_{i+1}" for i in range(len(req.unknowns))]
        if len(labels) != len(req.unknowns):
            raise HTTPException(status_code=400, detail="labels length must match unknowns length.")

        results = []
        for label, text_b in zip(labels, req.unknowns):
            r = compare(req.author_sample, text_b, use_embeddings=req.use_embeddings, quiet=True)
            results.append(_result_to_dict(r, label_a="author_sample", label_b=label))

        results.sort(key=lambda x: x["score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
        return results

    @app.post("/matrix")
    def api_matrix(req: MatrixRequest):
        """Compute N×N similarity matrix for a set of texts."""
        if len(req.texts) < 2:
            raise HTTPException(status_code=400, detail="Provide at least 2 texts.")
        if len(req.texts) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 texts per matrix request.")

        n = len(req.texts)
        labels = req.labels or [f"text_{i+1}" for i in range(n)]
        scores = [[0.0] * n for _ in range(n)]

        for i in range(n):
            scores[i][i] = 100.0
        for i in range(n):
            for j in range(i + 1, n):
                r = compare(req.texts[i], req.texts[j], use_embeddings=req.use_embeddings, quiet=True)
                scores[i][j] = r.score
                scores[j][i] = r.score

        return {"labels": labels, "matrix": scores}

    @app.post("/features")
    def api_features(req: FeaturesRequest):
        """Extract style features from a text."""
        try:
            feat = extract_features(req.text)
            return {
                "sent_length_mean": feat.sent_length_mean,
                "sent_length_std": feat.sent_length_std,
                "comma_per_sent": feat.comma_per_sent,
                "question_ratio": feat.question_ratio,
                "exclamation_ratio": feat.exclamation_ratio,
                "ellipsis_density": feat.ellipsis_density,
                "dash_density": feat.dash_density,
                "word_length_mean": feat.word_length_mean,
                "lexical_richness": feat.lexical_richness,
                "short_sent_ratio": feat.short_sent_ratio,
                "long_sent_ratio": feat.long_sent_ratio,
                "capitalized_mid_sent": feat.capitalized_mid_sent,
                "number_density": feat.number_density,
                "para_sent_mean": feat.para_sent_mean,
                "word_length_dist": feat.avg_word_length_dist,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return app
