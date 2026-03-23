"""Lightweight results cache for notebook figures.

Usage in notebook cells:

    from src.experiments.result_cache import cached

    @cached("fig3_nmse_vs_tau")
    def compute_fig3():
        # ... expensive computation ...
        return {"thresholds": [...], "nmses": [...]}

    data = compute_fig3()  # loads from JSON if cached, else computes and saves

Cache files are stored in assets/results/paper_cache/.
Delete a file to force recomputation.
"""
import json
import hashlib
from pathlib import Path
from functools import wraps
from typing import Callable, Any

import numpy as np

CACHE_DIR = Path("assets/results/paper_cache")


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": True, "data": obj.tolist(), "dtype": str(obj.dtype)}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _decode_numpy(obj):
    """JSON object hook that restores numpy arrays."""
    if isinstance(obj, dict) and "__ndarray__" in obj:
        return np.array(obj["data"], dtype=obj["dtype"])
    return obj


def cached(name: str, version: str = "v1"):
    """Decorator that caches function results as JSON.

    Args:
        name: cache file identifier (e.g., "fig3_nmse_vs_tau")
        version: bump to invalidate cache without deleting files
    """
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file = CACHE_DIR / f"{name}_{version}.json"

            if cache_file.exists():
                print(f"[cache] Loading {cache_file.name}")
                with open(cache_file) as f:
                    return json.load(f, object_hook=_decode_numpy)

            print(f"[cache] Computing {name}...")
            result = fn(*args, **kwargs)

            with open(cache_file, "w") as f:
                json.dump(result, f, cls=_NumpyEncoder, indent=2)
            print(f"[cache] Saved {cache_file.name} ({cache_file.stat().st_size / 1024:.1f} KB)")

            return result

        wrapper.invalidate = lambda: (CACHE_DIR / f"{name}_{version}.json").unlink(missing_ok=True)
        return wrapper
    return decorator


def invalidate_all():
    """Delete all cached results."""
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
        print(f"[cache] Cleared {CACHE_DIR}")
