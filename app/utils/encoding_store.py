"""Helper to safely save/load face encodings.

Provides:
- save_encodings_npz(output_path, encodings, ids)
- load_encodings(path) -> {'encodings': np.ndarray, 'ids': np.ndarray}

If a legacy .pkl file is present, `load_encodings` will migrate it to .npz and rename
the original to `<name>.pkl.migrated` to avoid re-running migration.
"""
from pathlib import Path
import pickle
import numpy as np
from typing import Dict, Any

from app.utils.logger import logger


def save_encodings_npz(output_path: Path | str, encodings, ids) -> Path:
    out_path = Path(output_path)
    if out_path.suffix != ".npz":
        out_path = out_path.with_suffix(".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enc_arr = np.array(encodings)
    ids_arr = np.array(ids)
    np.savez_compressed(out_path, encodings=enc_arr, ids=ids_arr)
    logger.info("Saved encodings to %s", out_path)
    return out_path


def _load_pkl_safe(pkl_path: Path) -> Dict[str, Any] | None:
    # Only for trusted/local files; we still perform validation after loading
    try:
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)
    except Exception as e:
        logger.error("Failed to load legacy pickle %s: %s", pkl_path, e)
        return None

    if not isinstance(data, dict):
        logger.error("Legacy pickle %s did not contain a dict", pkl_path)
        return None

    if "encodings" not in data or "ids" not in data:
        logger.error("Legacy pickle %s missing 'encodings' or 'ids' keys", pkl_path)
        return None

    return data


def load_encodings(path: Path | str) -> Dict[str, Any] | None:
    """Load encodings from a .npz file, migrating from .pkl if necessary.

    Returns a dict with 'encodings' and 'ids' as numpy arrays, or None on failure.
    """
    path = Path(path)

    # Prefer explicit .npz path
    npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
    pkl_path = path if path.suffix == ".pkl" else path.with_suffix(".pkl")

    if npz_path.exists():
        try:
            data = np.load(npz_path)
            enc = np.array(data["encodings"])
            ids = np.array(data["ids"])
            logger.debug("Loaded encodings from %s", npz_path)
            return {"encodings": enc, "ids": ids}
        except Exception as e:
            logger.error("Failed to load npz encodings from %s: %s", npz_path, e)
            return None

    # Attempt migration from legacy pickle
    if pkl_path.exists():
        logger.info("Found legacy pickle encodings at %s, attempting migration", pkl_path)
        legacy = _load_pkl_safe(pkl_path)
        if legacy is None:
            return None

        encodings = legacy.get("encodings")
        ids = legacy.get("ids")

        # Validate/convert
        try:
            enc_arr = np.array(encodings)
            ids_arr = np.array(ids)
        except Exception as e:
            logger.error("Failed to convert legacy encodings to numpy arrays: %s", e)
            return None

        # Save new .npz
        try:
            out_path = save_encodings_npz(npz_path, enc_arr, ids_arr)
            # Rename original pickle to indicate migration (don't delete automatically)
            migrated_path = pkl_path.with_suffix(pkl_path.suffix + ".migrated")
            pkl_path.rename(migrated_path)
            logger.info("Migrated legacy pickle %s -> %s", pkl_path, migrated_path)
            return {"encodings": enc_arr, "ids": ids_arr}
        except Exception as e:
            logger.error("Migration from pickle to npz failed: %s", e)
            return None

    logger.info("No encodings file found at %s (.npz or .pkl)", path)
    return None